import local_logger
import storage
import experiments
import llms
import strategies

from transformers.models.bert import BertTokenizerFast
from small_text.integrations.transformers.datasets import TransformersDataset
import datasets
import numpy as np
import numpy.typing as npt

import re
import dataclasses
import pathlib
import functools
import json
import itertools
from typing import Iterable, Callable
from collections import deque
from copy import deepcopy


class Indices(storage.Storable):
    def __init__(self, indices: npt.NDArray[np.int64]):
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)


@storage.Storable.make_storable(storage.StorableType.SEEDED_INDICES)
class SeededIndices(Indices):
    CURRENT_VERSION = 1

    def __init__(self, size: int, seed: int, dataset_size: int):
        indices = np.arange(dataset_size)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        super().__init__(indices=indices[:size])
        self.size = size
        self.seed = seed
        self.dataset_size = dataset_size

    @staticmethod
    def _get_salt(size: int, seed: int, dataset_size: int) -> storage.Hash:
        return storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, (size, seed, dataset_size))))

    def get_id(self) -> storage.ID:
        return SeededIndices.make_id(self.size, self.seed, self.dataset_size)

    @staticmethod
    def make_id(size: int, seed: int, dataset_size: int) -> storage.ID:
        return f'{seed}_{size}_{dataset_size}#{SeededIndices._get_salt(size, seed, dataset_size)}'

    def as_storable(self) -> storage.StorableBundle:
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'seed': self.seed,
                        'size': self.size,
                        'dataset_size': self.dataset_size,
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.SEEDED_INDICES,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < SeededIndices.CURRENT_VERSION:
            return SeededIndices.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: 'storage.StorableEntry', data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'SeededIndices':
        if entry.id in data and storable_type != storage.StorableType.SEEDED_INDICES:
            return data.retrieve(entry.id)
        payload, migrated = SeededIndices.migrate_to_newest_version(entry.payload)
        obj = SeededIndices(size=payload['size'], seed=payload['seed'], dataset_size=payload['dataset_size'])
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj


@storage.Stringifiable.make_stringifiable()
@dataclasses.dataclass(frozen=True, eq=True, slots=True)
class DatasetID(storage.Stringifiable):
    path: str
    subset: str | None = None

    STR_PATTERN = re.compile(r'(?P<path>[a-zA-Z0-9_]+(-[a-zA-Z0-9_]+)*)_(?P<subset>([a-zA-Z0-9_]+)?)')

    def __str__(self) -> str:
        return f'{self.path.replace('/', '-')}_{[self.subset, ''][self.subset is None]}'

    @staticmethod
    def from_str(id_str: str, db: 'DataDatabase') -> 'DatasetID':
        m = re.fullmatch(DatasetID.STR_PATTERN, id_str)
        if m is None:
            raise ValueError(f'Invalid DatasetID string: {id_str}')
        path = m.group('path').replace('-', '/')
        subset = m.group('subset') or None
        return DatasetID(path=path, subset=subset)


@storage.Storable.make_storable(storage.StorableType.LLM_LABELS)
@dataclasses.dataclass
class LLMLabels(storage.Storable):  # TODO: consider pool mapping
    dataset_id: DatasetID
    llm: llms.LLMType
    labels: npt.NDArray[np.int64]
    marked_up: npt.NDArray[np.bool]

    CURRENT_VERSION = 1

    def get_id(self) -> storage.ID:
        return f'{self.dataset_id}__{self.llm}#{self._get_salt(self.dataset_id, self.llm)}'

    @staticmethod
    def _get_salt(dataset_id: DatasetID, llm: llms.LLMType) -> storage.Hash:
        return storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, (dataset_id, llm))))

    def as_storable(self) -> storage.StorableBundle:
        salt = self._get_salt(self.dataset_id, self.llm)
        array_id = f'{salt}_1#{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, (salt, '1')))}'
        array_wrapper = ArrayWrapper(None, array_id, self.labels)
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'llm': str(self.llm),
                        'labels': array_wrapper.get_id(),
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.LLM_LABELS,
                    id=self.get_id(),
                ),
                **array_wrapper.as_storable().entries,
                # array_id: storage.StorableEntry.from_npy(self.labels),
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < LLMLabels.CURRENT_VERSION:
            return LLMLabels.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'LLMLabels':
        if entry.id in data and storable_type != storage.StorableType.LLM_LABELS:
            return data.retrieve(entry.id)
        payload, migrated = LLMLabels.migrate_to_newest_version(entry.payload)
        obj = LLMLabels(
            dataset_id=DatasetID.from_str(payload['dataset_id'], data),
            llm=llms.LLMType.from_str(payload['llm'], data),
            labels=data.retrieve(payload['labels']).array,
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj

    def __eq__(self, value):
        if not isinstance(value, LLMLabels):
            return False
        return (
            self.llm == value.llm and np.array_equal(self.labels, value.labels) and self.dataset_id == value.dataset_id
        )


class Dataset:
    def __init__(self, dataset: datasets.Dataset, base: 'CompleteDataset'):
        self.dataset = dataset
        self.base = base
        self.annotations = ...
        self.__y = None
        self.__x = None

    @property
    def y(self) -> npt.NDArray[np.int64]:
        if self.__y is None:
            self.__y = np.array(self.dataset['label'], dtype=np.int64)
        return self.__y

    @property
    def x(self) -> npt.NDArray[np.str_]:
        if self.__x is None:
            self.__x = np.array(self.dataset['text'], dtype=np.str_)
        return self.__x

    @property
    def size(self) -> int:
        return len(self.dataset)

    def __len__(self) -> int:
        return self.size

    @functools.cache
    def to_transformers_dataset(
        self, tokenizer: BertTokenizerFast, num_classes: int, max_length: int = 128
    ) -> TransformersDataset:
        return TransformersDataset.from_arrays(
            self.x,
            self.y,
            tokenizer,
            target_labels=np.arange(num_classes, dtype=np.int64),
            max_length=max_length,
        )


class LazyTrainDataset:
    def __init__(self, base: 'CompleteDataset'):
        self.base = base
        # self.annotations = ...
        self.__dataset = None

    @property
    def __internal_dataset(self) -> Dataset:
        if self.__dataset is None:
            self.__dataset = self.base.train
        return self.__dataset

    @property
    def dataset(self) -> datasets.Dataset:
        return self.__internal_dataset.dataset

    @property
    def y(self) -> npt.NDArray[np.int64]:
        return self.__internal_dataset.y

    @property
    def x(self) -> npt.NDArray[np.str_]:
        return self.__internal_dataset.x

    @property
    def size(self) -> int:
        return self.__internal_dataset.size

    def __len__(self) -> int:
        return len(self.__internal_dataset)

    def to_transformers_dataset(
        self, tokenizer: BertTokenizerFast, num_classes: int, max_length: int = 128
    ) -> TransformersDataset:
        return self.__internal_dataset.to_transformers_dataset(tokenizer, num_classes, max_length)


@storage.Storable.make_storable(storage.StorableType.POOL)
class Pool(storage.Storable):
    CURRENT_VERSION = 1

    def __init__(self, indices: Indices, dataset: Dataset):
        self.indices = indices
        self.dataset = dataset
        self.__y = None
        self.__x = None

    @property
    def y(self) -> npt.NDArray[np.int64]:
        if self.__y is None:
            self.__y = self.dataset.y[self.indices.indices]
        return self.__y

    @property
    def x(self) -> npt.NDArray[np.str_]:
        if self.__x is None:
            self.__x = self.dataset.x[self.indices.indices]
        return self.__x

    @property
    def subset(self) -> npt.NDArray[np.bool]:
        mask = np.zeros(self.dataset.x.shape[0], dtype=np.bool_)
        mask[self.indices.indices] = True
        return mask

    @property
    def base(self) -> 'CompleteDataset':
        return self.dataset.base

    @property
    def size(self) -> int:
        return len(self.indices)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _get_salt(dataset_id: DatasetID, indices_id: storage.ID) -> storage.Hash:
        return storage.Storable.combine_hashes(
            storage.Storable.hash_str(CompleteDataset.make_id(dataset_id)), storage.Storable.hash_str(indices_id)
        )

    def get_id(self) -> storage.ID:
        return self.make_id(self.base.id, self.indices.get_id())

    @staticmethod
    def make_id(dataset_id: DatasetID, indices_id: storage.ID) -> storage.ID:
        return f'{dataset_id}__{indices_id}#{Pool._get_salt(dataset_id, indices_id)}'

    def as_storable(self) -> storage.StorableBundle:
        indices_entry = self.indices.as_storable()
        dataset_entry = storage.StorableEntry(
            payload={
                'dataset_id': CompleteDataset.make_id(self.base.id),
                'indices': self.indices.get_id(),
                'version': self.CURRENT_VERSION,
            },
            type=storage.StorableType.POOL,
            id=self.get_id(),
        )
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): dataset_entry,
                **indices_entry.entries,
                **self.base.as_storable().entries,
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < Pool.CURRENT_VERSION:
            return Pool.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'Pool':
        if entry.id in data and storable_type != storage.StorableType.POOL:
            return data.retrieve(entry.id)
        payload, migrated = Pool.migrate_to_newest_version(entry.payload)
        pool = Pool(
            indices=data.retrieve(payload['indices']),
            dataset=data.retrieve(payload['dataset_id']).lazy_train,
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(pool)
        return pool

    @functools.cache
    def to_transformers_dataset(
        self, tokenizer: BertTokenizerFast, num_classes: int, max_length: int = 128
    ) -> TransformersDataset:
        return TransformersDataset.from_arrays(
            self.x,
            self.y,
            tokenizer,
            target_labels=np.arange(num_classes, dtype=np.int64),
            max_length=max_length,
        )

    def __repr__(self) -> str:
        return f'Pool(dataset_id={self.base.id}, indices={self.indices.indices})'

    # @property
    # def annotations(self) -> dict[LLMType, LLMLabels]:
    #     if self.__annotations is not None:
    #         return self.__annotations
    #     if self.dataset is not None:
    #         all_annotations = self.dataset.annotations
    #         self.__annotations = {}
    #         for llm, labels in all_annotations.items():
    #             pool_labels = labels.labels[self.indices.indices]
    #             self.__annotations[llm] = LLMLabels(llm=llm, labels=pool_labels)
    #         return self.__annotations
    #     raise ValueError("Cannot get annotations without base dataset")


@storage.Storable.make_storable(storage.StorableType.LLM_INDEX_RECAP_EXAMPLES)
@dataclasses.dataclass
class LLMIndexRecapExamples(storage.Storable):
    dataset_id: DatasetID
    subset: npt.NDArray[np.bool]
    llm: llms.LLMType
    recapped: dict[frozenset[int], npt.NDArray[np.int64]]
    step_size: int

    CURRENT_VERSION = 1

    @staticmethod
    def _get_salt(dataset_id: DatasetID, subset: npt.NDArray[np.bool], llm_type: llms.LLMType, step_size: int) -> str:
        return storage.Storable.combine_hashes(
            storage.Storable.hash_str(str(dataset_id)),
            storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0]))),
            storage.Storable.hash_str(str(llm_type)),
            storage.Storable.hash_str(str(step_size)),
        )

    @staticmethod
    def make_id(
        dataset_id: DatasetID, subset: npt.NDArray[np.bool], llm_type: llms.LLMType, step_size: int
    ) -> storage.ID:
        return f'{dataset_id}__{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0])))}__{llm_type}__{step_size}#{LLMIndexRecapExamples._get_salt(dataset_id, subset, llm_type, step_size)}'

    def get_id(self) -> storage.ID:
        return self.make_id(self.dataset_id, self.subset, self.llm, self.step_size)

    def as_storable(self) -> 'storage.StorableBundle':
        salt = self._get_salt(self.dataset_id, self.subset, self.llm, self.step_size)
        array_id = f'{salt}_1#{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, (salt, '1')))}'
        subset_array_wrapper = ArrayWrapper(None, array_id, self.subset)
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'dataset_id': str(self.dataset_id),
                        'llm': str(self.llm),
                        'subset': subset_array_wrapper.get_id(),
                        'step_size': str(self.step_size),
                        'recapped': [
                            {'recap': list(map(int, sorted(prev))), 'fetched': list(map(int, sorted(fetched)))}
                            for prev, fetched in self.recapped.items()
                        ],
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.LLM_INDEX_RECAP_EXAMPLES,
                    id=self.get_id(),
                ),
                **subset_array_wrapper.as_storable().entries,
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < LLMIndexRecapExamples.CURRENT_VERSION:
            return LLMIndexRecapExamples.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'LLMIndexRecapExamples':
        if entry.id in data and storable_type != storage.StorableType.LLM_INDEX_RECAP_EXAMPLES:
            return data.retrieve(entry.id)
        payload, migrated = LLMIndexRecapExamples.migrate_to_newest_version(entry.payload)
        obj = LLMIndexRecapExamples(
            DatasetID.from_str(payload['dataset_id'], data),
            data.retrieve(payload['subset']).array,
            llms.LLMType.from_str(payload['llm'], data),
            {
                frozenset(map(int, e['recap'])): np.array(sorted(map(int, e['fetched'])), dtype=np.int64)
                for e in payload['recapped']
            },
            int(payload['step_size']),
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj


@storage.Storable.make_storable(storage.StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS)
@dataclasses.dataclass
class DiversityBasedKMeansClusters(storage.Storable):
    dataset_id: DatasetID
    subset: npt.NDArray[np.bool]
    clustered: dict[int, tuple[npt.NDArray[np.int64], ...]]

    CURRENT_VERSION = 1

    @staticmethod
    def _get_salt(dataset_id: DatasetID, subset: npt.NDArray[np.bool]) -> str:
        return storage.Storable.combine_hashes(
            storage.Storable.hash_str('diversitybasedkmeansclusters'),
            storage.Storable.hash_str(str(dataset_id)),
            storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0]))),
        )

    @staticmethod
    def make_id(dataset_id: DatasetID, subset: npt.NDArray[np.bool]) -> storage.ID:
        return f'diversitybasedkmeansclusters__{dataset_id}__{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0])))}#{DiversityBasedKMeansClusters._get_salt(dataset_id, subset)}'

    def get_id(self) -> storage.ID:
        return self.make_id(self.dataset_id, self.subset)

    def as_storable(self) -> 'storage.StorableBundle':
        salt = self._get_salt(self.dataset_id, self.subset)
        array_id = f'{salt}_1#{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, (salt, '1')))}'
        subset_array_wrapper = ArrayWrapper(None, array_id, self.subset)
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'dataset_id': str(self.dataset_id),
                        'subset': subset_array_wrapper.get_id(),
                        'clustered': {
                            str(cluster_size): list(list(map(int, cluster)) for cluster in clusters)
                            for cluster_size, clusters in self.clustered.items()
                        },
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS,
                    id=self.get_id(),
                ),
                **subset_array_wrapper.as_storable().entries,
            },
        )

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload['version'] < DiversityBasedKMeansClusters.CURRENT_VERSION:
            assert False, 'impossible'
            # return ..., True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'DiversityBasedKMeansClusters':
        if entry.id in data and storable_type != storage.StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS:
            return data.retrieve(entry.id)
        payload, migrated = DiversityBasedKMeansClusters.migrate_to_newest_version(entry.payload)
        obj = DiversityBasedKMeansClusters(
            DatasetID.from_str(payload['dataset_id'], data),
            data.retrieve(payload['subset']).array,
            {
                int(cluster_size): tuple(np.array(cluster, dtype=np.int64) for cluster in clusters)
                for cluster_size, clusters in payload['clustered'].items()
            },
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj


@storage.Storable.make_storable(storage.StorableType.LLM_CLUSTER_EXAMPLES)
@dataclasses.dataclass
class LLMClusterExamples(storage.Storable):
    dataset_id: DatasetID
    subset: npt.NDArray[np.bool]
    llm: llms.LLMType
    cluster_to_examples: dict[frozenset[int], int]

    CURRENT_VERSION = 1

    @staticmethod
    def _get_salt(dataset_id: DatasetID, llm_type: llms.LLMType, subset: npt.NDArray[np.bool]) -> str:
        return storage.Storable.combine_hashes(
            'llmclusterexamples',
            storage.Storable.hash_str(str(dataset_id)),
            storage.Storable.hash_str(str(llm_type)),
            storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0]))),
        )

    @staticmethod
    def make_id(dataset_id: DatasetID, llm_type: llms.LLMType, subset: npt.NDArray[np.bool]) -> storage.ID:
        return f'llmclusterexamples__{dataset_id}__{llm_type}__{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, np.where(subset)[0])))}#{LLMClusterExamples._get_salt(dataset_id, llm_type, subset)}'

    def get_id(self) -> storage.ID:
        return self.make_id(self.dataset_id, self.llm, self.subset)

    def as_storable(self) -> 'storage.StorableBundle':
        salt = self._get_salt(self.dataset_id, self.llm, self.subset)
        array_id = f'{salt}_1#{storage.Storable.combine_hashes(*map(storage.Storable.hash_str, (salt, '1')))}'
        subset_array_wrapper = ArrayWrapper(None, array_id, self.subset)
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'dataset_id': str(self.dataset_id),
                        'subset': subset_array_wrapper.get_id(),
                        'llm': str(self.llm),
                        'cluster_to_examples': [
                            {'cluster': list(map(int, sorted(cluster))), 'fetched': selected}
                            for cluster, selected in self.cluster_to_examples.items()
                        ],
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.LLM_CLUSTER_EXAMPLES,
                    id=self.get_id(),
                ),
                **subset_array_wrapper.as_storable().entries,
            },
        )

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload['version'] < LLMClusterExamples.CURRENT_VERSION:
            assert False, 'impossible'
            # return ..., True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'LLMClusterExamples':
        if entry.id in data and storable_type != storage.StorableType.LLM_CLUSTER_EXAMPLES:
            return data.retrieve(entry.id)
        payload, migrated = LLMClusterExamples.migrate_to_newest_version(entry.payload)
        obj = LLMClusterExamples(
            DatasetID.from_str(payload['dataset_id'], data),
            data.retrieve(payload['subset']).array,
            llms.LLMType.from_str(payload['llm'], data),
            {frozenset(map(int, e['cluster'])): int(e['fetched']) for e in payload['cluster_to_examples']},
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj


@storage.Storable.make_storable(storage.StorableType.DATASET)
class CompleteDataset(storage.Storable):
    CURRENT_VERSION = 1

    def __init__(
        self,
        id: DatasetID,
        text_field: str,
        label_field: str,
        database: 'DataDatabase',
        *,
        train_len: None | int = None,
        cache_on_disk: bool = False,
    ):
        self.id = id
        self.text_field = text_field
        self.label_field = label_field
        self.__dataset = None
        self.__annotations = None
        self.__train = None
        self.__validation = None
        self.__database = database
        self.__train_len = train_len
        self.cache_on_disk = cache_on_disk

    @property
    def annotations(self) -> dict[llms.LLMType, LLMLabels]:  # TODO: pass storage object
        if self.__annotations is not None:
            return self.__annotations
        self.__annotations = {}
        dataset_file = self.__root_dir / storage.DATA_DIR / self.get_config_filename()
        if dataset_file.exists():
            with dataset_file.open('r') as df:
                config = json.load(df)
            self.__load_annotations(config)
        return self.__annotations

    def __load_annotations(self, config: dict):
        self.__annotations = {
            llms.LLMType.from_str(llm_str): LLMLabels.load(self.__root_dir, label_file)
            for llm_str, label_file in config['annotations'].items()
        }

    @property
    def train(self) -> Dataset:
        if self.__train is None:
            self.__train = Dataset(self.internal['train'], self)
        return self.__train

    @property
    def lazy_train(self) -> LazyTrainDataset:
        return LazyTrainDataset(self)

    @property
    def len_train(self):
        if self.__train_len is None:
            self.__train_len = len(self.train)
        return self.__train_len

    @property
    def validation(self) -> Dataset:
        if self.__validation is None:
            self.__validation = Dataset(self.internal.get('validation', self.internal['test']), self)
        return self.__validation

    @property
    def internal(self) -> datasets.DatasetDict:
        if self.__dataset is not None:
            return self.__dataset
        force = False
        try:
            if self.cache_on_disk and self.__cached_on_disk:
                self.__dataset = datasets.DatasetDict.load_from_disk(self.__dataset_dir)
                return self.__dataset
        except FileNotFoundError:
            force = True
        if self.id.subset is None:
            self.__dataset = datasets.load_dataset(self.id.path)
        else:
            self.__dataset = datasets.load_dataset(self.id.path, self.id.subset)
        self.__dataset = self.__standardize_dataset(self.__dataset)
        if self.cache_on_disk and (not self.__cached_on_disk or force):
            self.__dataset.save_to_disk(self.__dataset_dir)
        return self.__dataset

    @property
    def __cached_on_disk(self) -> bool:
        return self.__dataset_dir.exists()

    @property
    def __dataset_dir(self) -> pathlib.Path:
        return self.__database.root_dir / storage.DATA_DIR / 'bin' / 'datasets' / f'{self.get_id()}/'

    def pool(self, indices: Indices) -> Pool:
        id = Pool.make_id(self.id, indices.get_id())
        if id in self.__database:
            return self.__database.retrieve(id)
        obj = Pool(indices, self.lazy_train)
        self.__database.encache(obj)
        return obj

    @staticmethod
    def _get_salt(dataset_id: DatasetID) -> storage.Hash:
        return storage.Storable.hash_str(str(dataset_id))

    def get_id(self) -> storage.ID:
        return self.make_id(self.id)

    @staticmethod
    def make_id(dataset_id: DatasetID) -> storage.ID:
        return f'{dataset_id}#{CompleteDataset._get_salt(dataset_id)}'

    def as_storable(self) -> storage.StorableBundle:  # TODO: annotations
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'text_field': self.text_field,
                        'label_field': self.label_field,
                        'dataset': str(self.id),
                        'train_len': self.len_train,
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.DATASET,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < CompleteDataset.CURRENT_VERSION:
            return CompleteDataset.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'CompleteDataset':
        if entry.id in data and storable_type != storage.StorableType.DATASET:
            return data.retrieve(entry.id)
        payload, migrated = CompleteDataset.migrate_to_newest_version(entry.payload)
        obj = CompleteDataset(
            id=DatasetID.from_str(payload['dataset'], data),
            text_field=payload['text_field'],
            label_field=payload['label_field'],
            database=data,
            train_len=payload['train_len'],
        )
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj

    def __standardize_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        return dataset.map(
            lambda ex: {
                'text': ex[self.text_field],
                'label': ex[self.label_field],
            },
            remove_columns=dataset['train'].column_names,
        )


@storage.Storable.make_storable(storage.StorableType.DATASETS)
class Datasets(storage.Storable):
    CURRENT_VERSION = 1

    def __init__(self, *datasets: CompleteDataset):
        self.datasets = {dataset.id: dataset for dataset in datasets}

    def __contains__(self, key: CompleteDataset | DatasetID) -> bool:
        if isinstance(key, CompleteDataset):
            key = key.id
        return key in self.datasets

    def __delitem__(self, key: DatasetID):
        if key not in self.datasets:
            raise KeyError(f'Dataset with ID {key} not found')
        del self.datasets[key]

    def add(self, dataset: CompleteDataset) -> None:
        if dataset.id not in self.datasets:
            self.datasets[dataset.id] = dataset
            return
        raise ValueError('Dataset with the same storage.ID already exists in the collection')

    def __getitem__(self, key: DatasetID) -> CompleteDataset:
        if key in self.datasets:
            return self.datasets[key]
        raise KeyError('Dataset not found in the collection')

    def __iter__(self) -> Iterable[CompleteDataset]:
        return iter(self.datasets.values())

    @staticmethod
    def _get_salt() -> storage.Hash:
        return storage.Storable.hash_str('datasets')

    def get_id(self) -> storage.ID:
        return self.make_id()

    @staticmethod
    def make_id() -> storage.ID:
        return f'datasets#{Datasets._get_salt()}'

    def as_storable(self) -> storage.StorableBundle:
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'datasets': {
                            str(dataset_id): CompleteDataset.make_id(dataset.id)
                            for dataset_id, dataset in self.datasets.items()
                        },
                        'version': self.CURRENT_VERSION,
                    },
                    type=storage.StorableType.DATASETS,
                    id=self.get_id(),
                ),
                **dict(item for d in self.datasets.values() for item in d.as_storable().entries.items()),
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < Datasets.CURRENT_VERSION:
            return Datasets.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'Datasets':
        if entry.id in data and storable_type != storage.StorableType.DATASETS:
            return data.retrieve(entry.id)
        payload, migrated = Datasets.migrate_to_newest_version(entry.payload)
        obj = Datasets(data.retrieve(dataset_entry_id) for dataset_entry_id in payload['datasets'].values())
        if migrated:
            del data.stored_index[entry.id]
        data.encache(obj)
        return obj


@storage.Storable.make_storable(storage.StorableType.ARRAY_WRAPPER)
class ArrayWrapper(storage.Storable):
    CURRENT_VERSION = 1

    def __init__(self, database: 'DataDatabase', arr_id: storage.ID, arr: npt.NDArray | None = None):
        self.id = arr_id
        self.db = database
        self.__array = arr

    @property
    def array(self) -> npt.NDArray:
        if self.__array is None:
            self.__array = self.__load()
        return self.__array

    def __load(self):
        path = self.db.root_dir / storage.DATA_DIR / pathlib.Path(self.db.stored_index[self.id].obj.payload['rel_path'])
        return storage.Formatter.load(storage.Format.NPZ, path).payload['array']

    def _get_salt(self) -> storage.Hash:
        return storage.Storable.hash_str(self.id)

    def get_id(self) -> storage.ID:
        return self.id

    @staticmethod
    def make_id() -> storage.ID:
        raise ValueError('ArrayWrapper can\' have static id, it should be passed to concrete class')

    def as_storable(self) -> storage.StorableBundle:
        payload = {
            'rel_path': str(pathlib.Path('bin/') / storage.Format.NPZ.format_name(self.id, storage.StorableType.ARRAY)),
            'version': self.CURRENT_VERSION,
        }
        if self.__array is not None:
            payload['array'] = self.__array
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload=payload,
                    type=storage.StorableType.ARRAY_WRAPPER,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def migrate_payload_from_unversioned(payload: dict) -> dict:
        new_payload = deepcopy(payload)
        if 'version' not in payload:
            new_payload['version'] = 1

        return new_payload

    @staticmethod
    def migrate_to_newest_version(payload: dict) -> tuple[dict, bool]:
        if payload.get('version', 0) < ArrayWrapper.CURRENT_VERSION:
            return ArrayWrapper.migrate_payload_from_unversioned(payload), True
        return payload, False

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'ArrayWrapper':
        if entry.id in data and storable_type != storage.StorableType.ARRAY_WRAPPER:
            return data.retrieve(entry.id)
        obj = ArrayWrapper(data, entry.id)
        data.encache(obj)
        return obj


class DataDatabase:
    BACKUPS_COUNT = 5

    def __init__(
        self,
        root_dir: pathlib.Path,
        *,
        logger: local_logger.Logger | None = None,
        local: bool = True,
    ):  # TODO  implement remote
        self.objects_store: dict[storage.ID, storage.Storable] = {}
        self.stored_index: dict[storage.ID, storage.StoredEntry] = {}
        self.root_dir = root_dir
        self.local = local
        if logger is None:
            logger = local_logger.Logger(verbose_console=True)
        self.logger = logger
        self.__connected = False
        self.__datasets: Datasets = None
        self.__experiments: experiments.Experiments = None
        self.llms: dict[llms.LLMType, llms.LLM] = {}

    @property
    def datasets(self) -> Datasets:
        if self.__datasets is None:
            self.__datasets = Datasets()
        return self.__datasets

    @property
    def experiments(self) -> 'experiments.Experiments':
        if self.__experiments is None:
            self.__experiments = experiments.Experiments()
        return self.__experiments

    def connect_llm(self, llm_type: llms.LLMType, llm_instance: llms.LLM):
        if llm_type in self.llms:
            raise KeyError(f'LLM of type {llm_type} already stored in database!')
        self.llms[llm_type] = llm_instance

    def retrieve(self, obj_id: storage.ID, *, force_load: bool = False) -> storage.Storable:
        if obj_id in self.objects_store and not force_load:
            return self.objects_store[obj_id]
        if obj_id in self.stored_index:
            self.__recreate(self.stored_index[obj_id].obj)
            return self.objects_store[obj_id]
        raise KeyError(f'Object with id {obj_id} not found in the database')

    def encache(self, obj: storage.Storable, *, inconsistent_ok: bool = False):
        if obj.get_id() in self.objects_store:
            raise ValueError(f'Object with id {obj.get_id()} already exists in the cache')
        self.objects_store[obj.get_id()] = obj
        self.__on_cache_update(obj, inconsistent_ok=inconsistent_ok)

    def destroy(self, id: storage.ID, *, inconsistent_ok: bool = False):
        if id not in self.objects_store:
            raise KeyError(f'Object with id {id} not found in the database')
        obj = self.objects_store[id]
        del self.objects_store[id]
        if isinstance(obj, CompleteDataset):
            del self.datasets[obj.id]
        elif isinstance(obj, experiments.Experiment):
            del self.experiments[obj]

        if (
            isinstance(obj, (experiments.Experiment, experiments.ExperimentHistory))
            and obj in self.stored_index
            and not inconsistent_ok
        ):
            self.logger.warn(
                f'{'Experiment' if isinstance(obj, experiments.Experiment) else 'ExperimentHistory'} {obj} with id {id} deleted from objects, but present in stored_index'
            )

    def add_to_store_index(
        self,
        obj: storage.StorableBundle | storage.StorableEntry,
        *,
        force: bool = False,
        inconsistent_ok: bool | None = None,
        inconsistent_on_disk_ok: bool | None = None,
        inconsistent_in_cache_ok: bool | None = None,
    ):
        if inconsistent_ok is not None and (
            inconsistent_on_disk_ok is not None or inconsistent_in_cache_ok is not None
        ):
            raise ValueError(
                f'Incorrect parameter mix: {inconsistent_ok = }, {inconsistent_on_disk_ok = }, {inconsistent_in_cache_ok = }. First is incompatible with 2 other'
            )
        if inconsistent_ok is not None:
            inconsistent_on_disk_ok = inconsistent_ok
            inconsistent_in_cache_ok = inconsistent_ok
        if inconsistent_on_disk_ok is None:
            inconsistent_on_disk_ok = False
        if inconsistent_in_cache_ok is None:
            inconsistent_in_cache_ok = False

        if isinstance(obj, storage.StorableEntry):
            if force or obj.id not in self.stored_index:
                stored = False
                entry_format = None if obj.type != storage.StorableType.ARRAY else storage.Format.NPZ
                if (
                    not inconsistent_on_disk_ok
                    and obj.id in self.stored_index
                    and self.stored_index[obj.id].stored_on_disk
                ):
                    stored_entry = self.stored_index[obj.id]
                    loaded = self.__load_from_disk_presumably(obj.id, stored_entry.format, obj.type)
                    if self.storables_differ(loaded.obj, obj):
                        self.logger.warn(
                            f'for object with id {obj.id} stored entry on disk {loaded.obj} differs from entry being stored {obj}. Trying to merge...',
                            f'for object with id {obj.id} stored entry on disk differs from entry being stored. Trying to merge...',
                        )
                        try:
                            merged = self.merge_storables(obj, loaded.obj)
                        except ValueError:
                            self.logger.warn(
                                f'Failed to merge {obj.id}, preferring entry being stored {obj}, leaving disk one unchanged',
                                f'Failed to merge {obj.id}, preferring entry being stored, leaving disk one unchanged',
                            )
                            merged = self.__try_merge_storables(obj, loaded.obj)
                            obj = merged
                        else:
                            obj = merged
                            self.__store_entry(obj, stored_entry.format)
                            stored = True
                            entry_format = stored_entry.format
                self.stored_index[obj.id] = storage.StoredEntry(obj, stored=stored, format=entry_format)
                self.__on_storage_update(obj, inconsistent_ok=inconsistent_in_cache_ok)
            return
        if isinstance(obj, storage.StorableBundle):
            self.__add_to_store_index_rec(
                obj.main,
                obj,
                force=force,
                inconsistent_on_disk_ok=inconsistent_on_disk_ok,
                inconsistent_in_cache_ok=inconsistent_in_cache_ok,
            )
            return
        raise ValueError('Object must be either StorableEntry or StorableBundle')

    def __add_to_store_index_rec(
        self,
        key: storage.ID,
        bundle: storage.StorableBundle,
        *,
        force: bool = False,
        inconsistent_on_disk_ok: bool = False,
        inconsistent_in_cache_ok: bool = False,
    ):
        if key in self.stored_index and not force:
            return
        entry = bundle.entries[key]
        self.add_to_store_index(
            entry,
            force=force,
            inconsistent_on_disk_ok=inconsistent_on_disk_ok,
            inconsistent_in_cache_ok=inconsistent_in_cache_ok,
        )
        for ref in entry.get_references():
            self.__add_to_store_index_rec(
                ref,
                bundle,
                force=force,
                inconsistent_on_disk_ok=inconsistent_on_disk_ok,
                inconsistent_in_cache_ok=inconsistent_in_cache_ok,
            )

    def __on_cache_update(self, obj: storage.Storable, *, inconsistent_ok: bool = False):
        if isinstance(obj, CompleteDataset):
            if obj.id not in self.datasets:
                self.datasets.add(obj)
        elif isinstance(obj, experiments.Experiment):
            if obj not in self.experiments:
                self.experiments.add(obj)

        if not inconsistent_ok and obj.get_id() in self.stored_index:
            encached = obj.as_storable()
            encached = encached.entries[encached.main]
            was = self.stored_index[obj.get_id()].obj
            if DataDatabase.storables_differ(encached, was):
                try:
                    result = DataDatabase.merge_storables(encached, was)
                except ValueError:
                    self.logger.warn(
                        f'Can\'t merge storables {encached} and {was} with id {obj.get_id()}! Preferring encached {encached} for object {obj}',
                        f'Can\'t merge storables with id {obj.get_id()}! Preferring encached for object {type(obj)}',
                    )
                    result = DataDatabase.__try_merge_storables(encached, was)
                self.add_to_store_index(result, force=True)

    def __on_storage_update(self, obj: storage.StorableEntry, *, inconsistent_ok: bool = False):
        if (
            not inconsistent_ok
            or obj.type == storage.StorableType.EXPERIMENT
            or obj.type == storage.StorableType.DATASET
        ) and obj.id in self.objects_store:
            encached = self.objects_store[obj.id].as_storable()
            encached = encached.entries[encached.main]
            if DataDatabase.storables_differ(encached, obj):
                self.logger.warn(
                    f'object being stored {obj} differs from encached object {encached}. Trying to merge...',
                    f'object being stored differs from encached object. Trying to merge...',
                )
                try:
                    merged = self.merge_storables(encached, obj)
                except ValueError:
                    self.logger.warn(
                        f'Failed to merge {obj.id}, preferring the one just stored {obj}, differing from {encached}',
                        f'Failed to merge {obj.id}, preferring the one just stored, differing from encached',
                    )
                    merged = self.__try_merge_storables(obj, encached)
                self.destroy(merged.id, inconsistent_ok=True)
                self.add_to_store_index(merged, force=True, inconsistent_ok=True)
                if merged.id not in self.objects_store:
                    self.__recreate(merged)
        elif obj.type == storage.StorableType.EXPERIMENT or obj.type == storage.StorableType.DATASET:
            self.__recreate(obj)

    def __recreate(self, obj: storage.StorableEntry, *, inconsistent_after_encache_ok: bool = False):
        if obj.id in self.objects_store:
            raise ValueError(f'Trying to recreate existing object: {obj.id}')
        if obj.id not in self.stored_index:
            self.add_to_store_index(obj)
        assert obj.payload == self.stored_index[obj.id].obj.payload
        restored = storage.Storable.restore(obj, self, obj.type)
        if restored.get_id() not in self.objects_store:
            self.encache(restored, inconsistent_ok=inconsistent_after_encache_ok)
        return obj

    def store_fast(
        self, bundle: 'storage.StorableBundle', obj_format: storage.Format, obj_id: storage.ID | None = None
    ):
        if (
            obj_id is not None and obj_id in self.stored_index and not self.stored_index[obj_id].stored
        ):  # Secondary objects that aren't the main object being stored fast should not be stored fast and should only be updated if they are already stored
            return
        if obj_id is None:
            obj_id = bundle.main
        self.__store_entry(bundle.entries[obj_id], obj_format, update_index=False)
        for ref in bundle.entries[obj_id].get_references():
            self.store_fast(bundle, obj_format.switch_format(bundle.entries[ref].type), ref)
        if obj_id not in self.stored_index:
            self.stored_index[obj_id] = storage.StoredEntry(bundle.entries[obj_id], stored=True, format=obj_format)
            self.__on_storage_update(self.stored_index[obj_id].obj)
        else:
            if self.storables_differ(bundle.entries[obj_id], self.stored_index[obj_id].obj):
                self.logger.warn(
                    f'entry being stored fast {bundle.entries[obj_id]} differs from entry already stored! Trying to merge...',
                    f'entry being stored fast {obj_id} differs from entry already stored! Trying to merge...',
                )
                try:
                    merged = self.merge_storables(bundle.entries[obj_id], self.stored_index[obj_id].obj)
                except ValueError:
                    self.logger.warn(
                        f'Failed to merge, preferring entry being stored {bundle.entries[obj_id]} over {self.stored_index[obj_id]}',
                        f'Failed to merge, preferring entry being stored {obj_id} over one in stored index',
                    )
                    merged = self.__try_merge_storables(bundle.entries[obj_id], self.stored_index[obj_id].obj)
                self.add_to_store_index(merged, force=True)
                self.__store_entry(merged, obj_format)
            self.stored_index[obj_id].stored = True
            self.stored_index[obj_id].format = obj_format

    def __store_entry(self, obj: 'storage.StorableEntry', obj_format: storage.Format, *, update_index: bool = True):
        where = (
            self.root_dir
            / storage.DATA_DIR
            / self.__get_rel_directory(obj.type)
            / obj_format.format_name(obj.id, obj.type)
        )
        self.__pre_store_entry(obj, where)
        storage.Formatter.dump(obj, obj_format, where)
        self.__post_store_entry(obj, where, obj_format, update_index=update_index)

    def __get_rel_directory(self, entry_type: storage.StorableType) -> pathlib.Path:
        match entry_type:
            case storage.StorableType.ARRAY_WRAPPER:
                return pathlib.Path('wrappers')
            case storage.StorableType.ARRAY:
                return pathlib.Path('bin')
            case storage.StorableType.POOL:
                return pathlib.Path('markup', 'pools')
            case storage.StorableType.SEEDED_INDICES:
                return pathlib.Path('markup', 'indices')
            case storage.StorableType.EXPERIMENT_HISTORY:
                return pathlib.Path('experiments', 'histories')
            case storage.StorableType.DATASET:
                return pathlib.Path('datasets')
            case storage.StorableType.DATASETS | storage.StorableType.EXPERIMENTS:
                return pathlib.Path()
            case storage.StorableType.LLM_LABELS:
                assert False  # TODO: implement label storing
            case storage.StorableType.LLM_INDEX_RECAP_EXAMPLES:
                return pathlib.Path('llm', 'index_recap_examples')
            case storage.StorableType.LLM_CLUSTER_EXAMPLES:
                return pathlib.Path('llm', 'cluster_examples')
            case storage.StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS:
                return pathlib.Path('llm', 'diversity_based_k_means_clusters')
            case storage.StorableType.EXPERIMENT:
                return pathlib.Path('experiments')
            case _:
                assert False, 'unreachable'

    def __pre_store_entry(self, obj: 'storage.StorableEntry', where: pathlib.Path):
        where.parent.mkdir(parents=True, exist_ok=True)
        if obj.type == storage.StorableType.ARRAY_WRAPPER and 'array' in obj.payload:
            storage.Formatter.dump(
                storage.StorableEntry.from_npy(obj.payload['array'], None),
                storage.Format.NPZ,
                self.root_dir / storage.DATA_DIR / obj.payload['rel_path'],
            )
            del obj.payload['array']

    def __post_store_entry(
        self,
        obj: 'storage.StorableEntry',
        where: pathlib.Path,
        obj_format: 'storage.Format',
        *,
        update_index: bool = True,
    ):
        if update_index:
            self.stored_index[obj.id] = storage.StoredEntry(obj, stored=True, format=obj_format)
            self.__on_storage_update(self.stored_index[obj.id].obj)

    def dump(self):
        path = self.root_dir / storage.DATA_DIR / f'{self.get_config_name()}.json'
        last_backup_id = self.__pre_dump(path)
        current_backup_id = last_backup_id + 1
        if not self.local:
            assert False, 'TODO: implement remote'
        config = self.__generate_storables(current_backup_id)
        for entry in self.stored_index.values():
            if not entry.stored and not entry.obj.type.is_groupable():
                entry_format = storage.Format.JSON.switch_format(entry.obj.type)
                self.__store_entry(entry.obj, entry_format)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as dbf:
            json.dump(config, dbf)
        self.__post_dump(current_backup_id)

    def __pre_dump(self, new_config_path: pathlib.Path) -> int:
        if not self.local:
            assert False, 'TODO: implement remote'
        previous_backup_id = 0
        if new_config_path.exists():
            with new_config_path.open('r') as dbf:
                old_config = json.load(dbf)
            previous_backup_id = old_config.get('backup_id', 1)
            new_config_path.rename(new_config_path.parent / f'{self.get_config_name()}_{previous_backup_id + 1}.json')
        for obj in self.objects_store.values():
            encached = obj.as_storable()
            encached = encached.entries[encached.main]
            if obj.get_id() not in self.stored_index:
                self.add_to_store_index(obj.as_storable(), inconsistent_in_cache_ok=True)
            elif self.storables_differ(encached, self.stored_index[obj.get_id()].obj):
                self.logger.warn(
                    f'during dump for id {encached.id} inconsistent storable {self.stored_index[obj.get_id()].obj} and encached object {encached}! Preferring encached',
                    f'during dump for id {encached.id} inconsistent storable and encached object! Preferring encached',
                )
                self.add_to_store_index(obj.as_storable(), force=True)

        for obj in self.stored_index.values():
            if obj.obj.type == storage.StorableType.ARRAY_WRAPPER and 'array' in obj.obj.payload:
                del obj.obj.payload['array']

        return previous_backup_id

    def __post_dump(self, current_backup_id: int):
        if not self.local:
            assert False, 'TODO: implement remote'
        killable_old = (
            self.root_dir
            / storage.DATA_DIR
            / f'{self.get_config_name()}_{current_backup_id - DataDatabase.BACKUPS_COUNT + 1}.json'
        )  # Assume continious increasing backup ids
        if killable_old.exists():
            killable_old.unlink()

    def recollect_stored(self):
        recollected_files: list[pathlib.Path] = []
        tmp_dict = dict(self.stored_index)
        for obj_id, stored_entry in tmp_dict.items():
            if stored_entry.stored_on_disk and stored_entry.obj.type.is_groupable():
                path = (
                    self.root_dir
                    / storage.DATA_DIR
                    / self.__get_rel_directory(stored_entry.obj.type)
                    / stored_entry.format.format_name(obj_id, stored_entry.obj.type)
                )
                loaded = self.__load_from_disk_by_path(obj_id, path)
                assert obj_id in self.stored_index
                if obj_id in self.objects_store:
                    encached = self.objects_store[obj_id].as_storable()
                    encached = encached.entries[encached.main]
                    assert not self.storables_differ(encached, self.stored_index[obj_id].obj)
                    if self.storables_differ(stored_entry.obj, loaded.obj):
                        try:
                            merged = self.merge_storables(stored_entry.obj, loaded.obj)
                        except ValueError:
                            self.logger.error(
                                f'Failed to recollect object with id {obj_id}. Skipping {path}. On disk: {loaded.obj}, in RAM: {stored_entry.obj}.',
                                f'Failed to recollect object with id {obj_id}. Skipping {path}',
                            )
                        else:
                            self.add_to_store_index(merged, force=True, inconsistent_on_disk_ok=True)
                            recollected_files.append(path)
                            self.stored_index[obj_id].stored = False
                            self.stored_index[obj_id].format = None
                    else:
                        recollected_files.append(path)
                        self.stored_index[obj_id].stored = False
                        self.stored_index[obj_id].format = None
                else:
                    if self.storables_differ(stored_entry.obj, loaded.obj):
                        try:
                            merged = self.merge_storables(stored_entry.obj, loaded.obj)
                        except ValueError:
                            self.logger.error(
                                f'Failed to recollect object with id {obj_id}. Skipping {path}. On disk: {loaded.obj}, in RAM: {stored_entry.obj}.',
                                f'Failed to recollect object with id {obj_id}. Skipping {path}',
                            )
                        else:
                            self.add_to_store_index(merged, force=True, inconsistent_ok=True)
                            recollected_files.append(path)
                            self.stored_index[obj_id].stored = False
                            self.stored_index[obj_id].format = None
                    else:
                        recollected_files.append(path)
                        self.stored_index[obj_id].stored = False
                        self.stored_index[obj_id].format = None

        self.dump()

        for file in recollected_files:
            self.logger.warn(f'deleting file {file}')
            file.unlink()

        def warn_unindexed(folder: pathlib.Path, file: pathlib.Path):
            stored_type = storage.Format.type_from_format_name(file.name)
            if not stored_type.is_groupable():
                return
            stored_id = storage.Format.id_from_format_name(file.name)
            if stored_id not in self.stored_index:
                self.logger.warn(f'unindexed file: {file} for id {stored_id} with type {stored_type}')

        self.__walk_local_storage(warn_unindexed)

    def try_restore(self):
        def completely_storable(obj: storage.StorableEntry) -> bool:
            # if obj.type == storage.StorableType.EXPERIMENT:
            #     assert False, "TODO: remove histories that are missing"
            for ref_id in obj.get_references():
                if ref_id not in self:
                    return False
            return True

        def process_file(folder: pathlib.Path, file: pathlib.Path):
            stored_id = storage.Format.id_from_format_name(file.name)
            loaded = self.__load_from_disk_by_path(stored_id, self.root_dir / storage.DATA_DIR / file)

            is_encached = stored_id in self.objects_store
            is_stored = stored_id in self.stored_index

            stored_entry = None if not is_stored else self.stored_index[stored_id].obj
            was = None if not is_encached else self.objects_store[stored_id]

            if not is_stored and not is_encached:
                if not completely_storable(loaded.obj):
                    ref_id = next(ref_id for ref_id in loaded.obj.get_references() if ref_id not in self)
                    self.logger.error(
                        f'Failed to load {file}, because referenced id {ref_id} is missing. Loaded: {loaded.obj}',
                        f'Failed to load {file}, because referenced id {ref_id} is missing',
                    )
                    return
                self.add_to_store_index(loaded.obj)
            elif not is_stored and is_encached:
                self.logger.warn(f'unindexed stored file: {file} that was recreated: {stored_id}')
                bundle = was.as_storable()
                entry = bundle.entries[bundle.main]
                try:
                    merged = DataDatabase.merge_storables(entry, loaded.obj)
                except ValueError:
                    self.logger.error(
                        f'Failed to merge id {entry.id} loaded {loaded.obj} with encached {stored_entry}. Trying to merge with encached preference...',
                        f'Failed to merge id {entry.id} loaded with encached. Trying to merge with encached preference...',
                    )
                    merged = DataDatabase.__try_merge_storables(entry, loaded.obj)
                if not completely_storable(merged):
                    self.logger.error(
                        f'Failed to add merged id {merged.id} entry {merged} when merging with {entry}. Skipping...',
                        f'Failed to add merged id {merged.id} entry when merging with loaded. Skipping...',
                    )
                    return
                self.destroy(merged.id)
                self.__recreate(merged)
            elif is_stored and not is_encached:
                if self.storables_differ(stored_entry, loaded.obj):
                    try:
                        merged = DataDatabase.merge_storables(loaded.obj, stored_entry)
                    except ValueError:
                        self.logger.error(
                            f'Failed to merge id {stored_entry.id} loaded {loaded.obj} with stored {stored_entry}. Trying to merge with stored preference...',
                            f'Failed to merge id {stored_entry.id} loaded with stored. Trying to merge with stored preference...',
                        )
                        merged = DataDatabase.__try_merge_storables(stored_entry, loaded.obj)
                    if not completely_storable(merged):
                        self.logger.error(
                            f'Failed to add merged id {merged.id} entry {merged} when merging with {stored_entry}. Skipping...',
                            f'Failed to add merged id {merged.id} entry when merging with stored. Skipping...',
                        )
                        return
                    self.add_to_store_index(merged, force=True)
                elif not self.stored_index[stored_id].stored_on_disk:
                    self.logger.warn(f'Consistent stored on disk file: {file}. Marking as stored on disk')
                    self.stored_index[stored_id].stored = True
                    self.stored_index[stored_id].format = storage.Format.format_from_format_name(file.name)
            elif is_stored and is_encached:
                bundle = was.as_storable()
                entry = bundle.entries[bundle.main]
                assert not self.storables_differ(entry, stored_entry)
                if self.storables_differ(stored_entry, loaded.obj):
                    try:
                        merged = DataDatabase.merge_storables(stored_entry, loaded.obj)
                    except ValueError:
                        self.logger.error(
                            f'Failed to merge id {stored_entry.id} loaded {loaded.obj} with stored {stored_entry}. Trying to merge with in db preference...',
                            f'Failed to merge id {stored_entry.id} loaded with stored. Trying to merge with in db preference...',
                        )
                        merged = DataDatabase.__try_merge_storables(stored_entry, loaded.obj)
                    if not completely_storable(merged):
                        self.logger.error(
                            f'Failed to add merged id {merged.id} entry {merged} when merging with {stored_entry}. Skipping...',
                            f'Failed to add merged id {merged.id} entry when merging with stored. Skipping...',
                        )
                        return
                    self.destroy(merged.id)
                    self.add_to_store_index(merged, force=True)
                    if not self.stored_index[stored_id].stored_on_disk:
                        self.__store_entry(
                            self.stored_index[stored_id].obj,
                            self.stored_index[stored_id].format or storage.Format.JSON,
                        )
                elif not self.stored_index[stored_id].stored_on_disk:
                    self.logger.warn(f'Consistent stored on disk file: {file}. Marking as stored on disk')
                    self.stored_index[stored_id].stored = True
                    self.stored_index[stored_id].format = storage.Format.format_from_format_name(file.name)
            else:
                assert False, 'unreachable'

        self.__walk_local_storage(process_file)

    def unroll(self):
        for obj in self.objects_store.values():
            if obj.get_id() not in self.stored_index:
                self.add_to_store_index(obj.as_storable())
        for stored_entry in self.stored_index.values():
            if not stored_entry.stored_on_disk:
                self.__store_entry(stored_entry.obj, storage.Format.JSON)

    def __walk_local_storage(self, callback: Callable[[pathlib.Path, pathlib.Path], None]):
        data_folder = self.root_dir / storage.DATA_DIR
        for folder in (
            data_folder / 'wrappers',
            data_folder / 'llm' / 'index_recap_examples',
            data_folder / 'datasets',
            data_folder / 'markup' / 'indices',
            data_folder / 'markup' / 'pools',
            data_folder / 'experiments' / 'histories',
            data_folder / 'experiments',
            data_folder,
        ):
            if not folder.exists():
                continue
            for file in folder.iterdir():
                if file.name == '.DS_Store':
                    continue
                if file.is_file() and not file.name.startswith(self.get_config_name()):
                    callback(folder, file.relative_to(data_folder))

    def __contains__(self, obj: 'storage.ID | experiments.Experiment | DatasetID') -> bool:
        if isinstance(obj, storage.ID):
            return obj in self.objects_store or obj in self.stored_index
        if isinstance(obj, experiments.Experiment):
            return obj in self.experiments
        elif isinstance(obj, DatasetID):
            return obj in self.datasets
        return NotImplemented

    def get_dataset(
        self, obj: DatasetID, text_field: str = 'text', label_field: str = 'label', cache: bool = False
    ) -> CompleteDataset:
        if obj in self.datasets:
            self.datasets.datasets[obj].cache_on_disk = cache
            return self.datasets.datasets[obj]
        dataset = CompleteDataset(
            id=obj,
            text_field=text_field,
            label_field=label_field,
            database=self,
            cache_on_disk=cache,
        )
        self.encache(dataset)

        return dataset

    def get_experiment(self, obj: 'experiments.Experiment') -> 'experiments.Experiment':
        if obj.get_id() in self.stored_index and obj.get_id() not in self.objects_store:
            raise RuntimeError(
                f'Inconsistent stored_index and objects_store for Experiment; id: {obj.get_id()}, experiment: {self.stored_index[obj.get_id()]}'
            )
        if obj in self.experiments:
            return self.experiments[obj][0]
        self.encache(obj)
        return obj

    def get_llm(self, llm_type: llms.LLMType) -> llms.LLM:
        if llm_type not in self.llms:
            raise KeyError(f'LLM of type {llm_type} is not connected!')
        return self.llms[llm_type]

    @staticmethod
    def get_config_name() -> str:
        return 'DONT_EVER_TOUCH_THIS_FILE_database8=D'

    def __generate_storables(self, backup_id) -> dict:
        return {
            'backup_id': backup_id,
            'objects': {
                obj_id: {
                    'type': entry.obj.type.value,
                    'payload': entry.obj.payload,
                }
                for obj_id, entry in self.stored_index.items()
                if not entry.stored_on_disk and entry.obj.type.is_groupable()
            },
            'refs': {
                obj_id: {
                    'rel_path': str(  # TODO: rename everywhere and check `rel_path`
                        self.__get_rel_directory(entry.obj.type) / entry.format.format_name(obj_id, entry.obj.type)
                    ),
                    'format': entry.format.value,
                    'type': entry.obj.type.value,
                    # TODO: consider remote
                }
                for obj_id, entry in self.stored_index.items()
                if entry.stored_on_disk
                or (
                    not entry.obj.type.is_groupable()
                    and (
                        self.logger.warn(
                            f'ungrouppable entry {entry.obj.id} is not stored on disk after pre dump during storables list generation! Entry contents: {entry}',
                            f'ungrouppable entry {entry.obj.id} is not stored on disk after pre dump during storables list generation!',
                        )
                        and True
                    )
                )
            },
            'experiments': [exp.get_id() for exp in self.experiments],
            'datasets': [dataset.get_id() for dataset in self.datasets],
        }

    @classmethod
    def load_default_config_name(cls, root_dir: pathlib.Path, *, local: bool = True) -> 'DataDatabase':
        return cls.load(root_dir / storage.DATA_DIR / f'{cls.get_config_name()}.json', root_dir, local=local)

    @classmethod
    def load(
        cls,
        config_path: pathlib.Path,
        root_dir: pathlib.Path,
        *,
        logger: local_logger.Logger | None = None,
        local: bool = True,
    ) -> 'DataDatabase':
        if not local:
            assert False, 'TODO: implement remote'
        if not config_path.exists():
            raise FileNotFoundError(f'Config file {config_path} does not exist')
        with config_path.open('r') as dbf:
            config = json.load(dbf)
        db = cls(root_dir, logger=logger, local=local)
        db.__load_config(config)
        return db

    def __load_config(self, config: dict):
        for obj_id, obj in config['objects'].items():
            if (
                storage.StorableType(obj['type']) == storage.StorableType.EXPERIMENT
                and obj_id not in config['experiments']
            ):
                config['experiments'].append(obj_id)

        dead_ids: deque[tuple[str, storage.StorableType]] = deque()
        for obj_id, obj in config['objects'].items():
            self.stored_index[obj_id] = storage.StoredEntry(
                obj=storage.StorableEntry(
                    payload=obj['payload'],
                    type=storage.StorableType(obj['type']),
                    id=obj_id,
                ),
                stored=False,
                format=None,
            )
            for ref_id, ref_type in storage.StorableEntry.get_references_from_pythonish(
                storage.StorableType(obj['type']), obj['payload']
            ):
                if not any(ref_id in config[value] for value in ['objects', 'refs', 'datasets']):
                    dead_ids.append((ref_id, ref_type))

        for obj_id in config['refs']:
            try:
                self.stored_index[obj_id] = self.__load_from_disk(
                    config, obj_id
                )  # TODO: consider adding refs not to load everything on start
            except FileNotFoundError:
                self.logger.warn(f'file for stored object with id {obj_id} not found, considering it dead id')
                dead_ids.append((obj_id, storage.StorableType(config['refs'][obj_id]['type'])))

        self.__recursively_clean_dead_ids(config, dead_ids)

        assert not self.objects_store
        assert not tuple(self.experiments)
        assert not tuple(self.datasets)

        self.__migrate_entries_on_disk(config)

        for obj_id in itertools.chain(config['datasets'], config['experiments']):
            if obj_id not in self.objects_store and obj_id in self.stored_index:
                self.__recreate(self.stored_index[obj_id].obj, inconsistent_after_encache_ok=True)

    def __migrate_entries_on_disk(self, config: dict):
        tmp_dict = dict(self.stored_index.items())
        for entry_id, entry in tmp_dict.items():
            if entry.obj.type == storage.StorableType.ARRAY_WRAPPER:
                continue

            entry_cls = storage.Storable.storable_classes[storage.StorableType(entry.obj.type)]
            if entry.obj.payload.get('version', 0) < entry_cls.CURRENT_VERSION:
                self.logger.warn(
                    f'entry with id {entry_id} migrating from format version {entry.obj.payload.get('version', 0)} to {entry_cls.CURRENT_VERSION}. Previous payload: {entry.obj.payload}. Changing stored index!',
                    f'entry with id {entry_id} migrating from format version {entry.obj.payload.get('version', 0)} to {entry_cls.CURRENT_VERSION}. Changing stored index!',
                )
                migrated_payload = entry_cls.migrate_to_newest_version(entry.obj.payload)[0]
                entry.obj = dataclasses.replace(entry.obj, payload=migrated_payload)
            entry = self.stored_index[entry_id]

            if entry.stored_on_disk:
                loaded = self.__load_from_disk(config, entry_id)
                if loaded.obj.payload.get('version', 0) < entry_cls.CURRENT_VERSION:
                    self.logger.warn(
                        f'entry with id {entry_id} migrating from format version {loaded.obj.payload.get('version', 0)} to {entry_cls.CURRENT_VERSION}. Previous payload: {loaded.obj.payload}. Changing stored on disk!',
                        f'entry with id {entry_id} migrating from format version {loaded.obj.payload.get('version', 0)} to {entry_cls.CURRENT_VERSION}. Changing stored on disk!',
                    )
                    self.__store_entry(entry.obj, entry.format, update_index=False)

    def __load_from_disk(self, config: dict, obj_id: 'storage.ID') -> 'storage.StoredEntry':
        if not self.local:
            assert False, 'TODO: implement remote'

        if obj_id not in config['refs']:
            raise KeyError(f'Object with id {obj_id} not found in config')
        content = config['refs'][obj_id]
        obj_format = storage.Format(content['format'])
        path: pathlib.Path = self.root_dir / storage.DATA_DIR / content['rel_path']
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist for object with id {obj_id}')
        entry = storage.Formatter.load(obj_format, path)
        return storage.StoredEntry(obj=entry, stored=True, format=obj_format)

    def __load_from_disk_presumably(
        self, id: 'storage.ID', entry_format: storage.Format, entry_type: storage.StorableType
    ) -> 'storage.StoredEntry':
        if not self.local:
            assert False, 'TODO: implement remote'

        path: pathlib.Path = (
            self.root_dir
            / storage.DATA_DIR
            / self.__get_rel_directory(entry_type)
            / entry_format.format_name(id, entry_type)
        )
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist for object with id {id}')
        entry = storage.Formatter.load(entry_format, path)
        return storage.StoredEntry(obj=entry, stored=True, format=entry_format)

    def __load_from_disk_by_path(self, id: 'storage.ID', path: pathlib.Path) -> 'storage.StoredEntry':
        if not self.local:
            assert False, 'TODO: implement remote'

        entry_format = storage.Format.format_from_format_name(path.name)

        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist for object with id {id}')
        entry = storage.Formatter.load(entry_format, path)
        return storage.StoredEntry(obj=entry, stored=True, format=entry_format)

    def __recursively_clean_dead_ids(self, config: dict, dead_ids: deque[tuple[str, storage.StorableType]]):
        to_be_died = set(map(lambda x: x[0], dead_ids))
        while dead_ids:
            dead_id, dead_type = dead_ids.popleft()

            tmp = dict(self.stored_index)
            for stored_id, entry in tmp.items():
                if self.stored_index[stored_id].obj.type in (
                    storage.StorableType.EXPERIMENTS,
                    storage.StorableType.DATASETS,
                    storage.StorableType.LLM_LABELS,
                ):  # Assumes everything is already loaded into stored_index
                    continue
                if dead_id in entry.obj.get_references():
                    if (
                        dead_type == storage.StorableType.EXPERIMENT_HISTORY
                        and entry.obj.type == storage.StorableType.EXPERIMENT
                    ):
                        self.logger.warn(
                            f'History {dead_id} for experiment {entry.obj.id} not found, popping from histories'
                        )
                        run_number = (
                            list(
                                map(
                                    lambda x: x[1],
                                    sorted(entry.obj.payload['histories'].items(), key=lambda p: int(p[0])),
                                )
                            ).index(dead_id)
                            + 1
                        )
                        runs = entry.obj.payload['runs']
                        del entry.obj.payload['histories'][str(run_number)]
                        for i in range(run_number + 1, runs + 1):
                            entry.obj.payload['histories'][str(i - 1)] = entry.obj.payload['histories'][str(i)]
                        if run_number != runs:
                            del entry.obj.payload['histories'][str(runs)]
                        entry.obj.payload['runs'] -= 1
                        if entry.stored_on_disk:
                            path = (
                                self.root_dir
                                / storage.DATA_DIR
                                / self.__get_rel_directory(entry.obj.type)
                                / entry.format.format_name(entry.obj.id, entry.obj.type)
                            )
                            self.logger.warn(f'Updating stored file {path} with cutted version')
                            self.__store_entry(entry.obj, entry.format, update_index=False)
                    else:
                        self.logger.warn(
                            f'incomplete object {stored_id} removing from stored index. Content: {entry.obj}',
                            f'incomplete object {stored_id} removing from stored index',
                        )
                        stored_type = self.stored_index[stored_id].obj.type
                        del self.stored_index[stored_id]
                        if stored_id not in to_be_died:
                            dead_ids.append((stored_id, stored_type))
                            to_be_died.add(stored_id)

    # def connect(self) -> "DataDatabase":
    #     if self.__connected:
    #         raise RuntimeError("Database is already connected, cannot connect again")
    #     self.__connected = True
    #     if not self.local:  # TODO: implement remote
    #         raise NotImplementedError("Remote databases are not implemented yet")
    #     if (self.root_dir / storage.DATA_DIR / f"{DataDatabase.get_config_name()}.json").exists():
    #         self.load()
    #     return self

    @staticmethod
    def storables_differ(obj1: 'storage.StorableEntry', obj2: 'storage.StorableEntry') -> bool:
        if obj1.type != obj2.type:
            raise ValueError('Can\'t compare different types')

        payload1 = storage.Storable.storable_classes[storage.StorableType(obj1.type)].migrate_to_newest_version(
            obj1.payload
        )[0]
        payload2 = storage.Storable.storable_classes[storage.StorableType(obj2.type)].migrate_to_newest_version(
            obj2.payload
        )[0]

        if obj1.type == storage.StorableType.ARRAY_WRAPPER:

            def simplify(payload: dict) -> dict:
                keys = set(payload) - {'array'}

                return {key: payload[key] for key in keys}

            return simplify(payload1) != simplify(payload2)
        return payload1 != payload2

    @staticmethod
    def merge_storables(obj1: 'storage.StorableEntry', obj2: 'storage.StorableEntry') -> 'storage.StorableEntry':
        res1 = DataDatabase.__try_merge_storables(obj1, obj2)
        res2 = DataDatabase.__try_merge_storables(obj2, obj1)
        if res1.payload != res2.payload:
            raise ValueError(
                f'Inconsistent merging! ids: {res1.id}, {res2.id}, type: {res1.type}, merge results: {res1.payload}, {res2.payload}'
            )

        return res1

    @staticmethod
    def __try_merge_storables(
        main: 'storage.StorableEntry', secondary: 'storage.StorableEntry'
    ) -> 'storage.StorableEntry':
        if main.type != secondary.type:
            raise ValueError('Can\'t merge different types')
        if main.type == storage.StorableType.EXPERIMENT:
            payload = deepcopy(main.payload)
            l1 = len(payload['histories'])
            l2 = len(secondary.payload['histories'])
            if l2 > l1:
                payload['histories'].update(
                    {i: secondary.payload['histories'][i] for i in map(str, range(l1 + 1, l2 + 1))}
                )
            if not all(
                secondary.payload['histories'][i] in payload['histories'].values()
                for i in map(str, range(1, min(l1, l2) + 1))
            ):
                j = max(l1, l2) + 1
                for hist in secondary.payload['histories'].values():
                    if hist not in payload['histories'].values():
                        payload['histories'][str(j)] = hist
                        j += 1
            payload['runs'] = len(payload['histories'])
            payload['histories'] = dict(
                map(lambda x: (str(x[0]), x[1]), enumerate(sorted(payload['histories'].values()), start=1))
            )
            return storage.StorableEntry(payload, main.type, main.id)
        if main.type == storage.StorableType.LLM_INDEX_RECAP_EXAMPLES:

            def simplify(payload: dict) -> dict:
                keys = set(payload) - {'array'}
                return {key: payload[key] for key in keys}

            payload = simplify(main.payload)
            ks1 = set(tuple(entry['recap']) for entry in payload['recapped'])
            ks2 = set(tuple(entry['recap']) for entry in secondary.payload['recapped'])
            if ks2 != ks1:
                payload['recapped'].extend(
                    entry for entry in secondary.payload['recapped'] if tuple(entry['recap']) in ks2 - ks1
                )
            if 'array' in main.payload:
                payload['array'] = main.payload['array']
            return storage.StorableEntry(payload, main.type, main.id)
        if main.type == storage.StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS:

            def simplify(payload: dict) -> dict:
                keys = set(payload) - {'array'}
                return {key: payload[key] for key in keys}

            payload = simplify(main.payload)
            ks1 = set(entry for entry in payload['clustered'])
            ks2 = set(entry for entry in secondary.payload['clustered'])
            if ks2 != ks1:
                payload['clustered'].update(
                    (key, entry) for key, entry in secondary.payload['clustered'].items() if key in ks2 - ks1
                )
            if 'array' in main.payload:
                payload['array'] = main.payload['array']
            return storage.StorableEntry(payload, main.type, main.id)
        if main.type == storage.StorableType.LLM_CLUSTER_EXAMPLES:

            def simplify(payload: dict) -> dict:
                keys = set(payload) - {'array'}
                return {key: payload[key] for key in keys}

            payload = simplify(main.payload)
            ks1 = set(tuple(entry['cluster']) for entry in payload['cluster_to_examples'])
            ks2 = set(tuple(entry['cluster']) for entry in secondary.payload['cluster_to_examples'])
            if ks2 != ks1:
                payload['cluster_to_examples'].extend(
                    entry for entry in secondary.payload['cluster_to_examples'] if tuple(entry['cluster']) in ks2 - ks1
                )
            if 'array' in main.payload:
                payload['array'] = main.payload['array']
            return storage.StorableEntry(payload, main.type, main.id)
        raise ValueError(f'Unknown types for merge: {main.type}')

    def __enter__(self):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError('Remote databases are not implemented yet')

        config_path = self.root_dir / storage.DATA_DIR / f'{self.get_config_name()}.json'
        if config_path.exists():
            tmp = DataDatabase.load(config_path, self.root_dir, logger=self.logger, local=self.local)
            self.objects_store = tmp.objects_store
            self.stored_index = tmp.stored_index
            self.__datasets = tmp.__datasets
            self.__experiments = tmp.__experiments
            self.llms = tmp.llms
            for exp in self.__experiments:
                if isinstance(
                    exp.cold_start_strategy.query_strategy, strategies.ActiveLLMQueryStrategyType
                ) or isinstance(exp.cold_start_strategy.query_strategy, strategies.SelectLLMQueryStrategyType):
                    exp.cold_start_strategy.query_strategy.set_db(self)
                if isinstance(
                    exp.active_learning_strategy.query_strategy, strategies.ActiveLLMQueryStrategyType
                ) or isinstance(exp.active_learning_strategy.query_strategy, strategies.SelectLLMQueryStrategyType):
                    exp.active_learning_strategy.query_strategy.set_db(self)
            for obj in self.objects_store:
                if isinstance(obj, strategies.ActiveLLMQueryStrategyType) or isinstance(
                    obj, strategies.SelectLLMQueryStrategyType
                ):
                    obj.set_db(self)
            tmp.objects_store = {}
            tmp.stored_index = {}
            tmp.__datasets = None
            tmp.__experiments = None
            tmp.llms = None
        self.__connected = True
        assert self.llms is not None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError('Remote databases are not implemented yet')
        self.dump()
        self.__connected = False
