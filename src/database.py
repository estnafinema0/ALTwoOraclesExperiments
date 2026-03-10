from utils import EnumABCMeta
import storage
import experiments
from transformers.models.bert import BertTokenizerFast
from small_text.integrations.transformers.datasets import TransformersDataset
import datasets
import numpy as np
import numpy.typing as npt

import re
import dataclasses
import pathlib
import functools
import enum
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
                    },
                    type=storage.StorableType.SEEDED_INDICES,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def from_storable(
        entry: 'storage.StorableEntry', data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'SeededIndices':
        if entry.id in data and storable_type != storage.StorableType.SEEDED_INDICES:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = SeededIndices(size=payload['size'], seed=payload['seed'], dataset_size=payload['dataset_size'])
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
    def from_str(id_str: str) -> 'DatasetID':
        m = re.fullmatch(DatasetID.STR_PATTERN, id_str)
        if m is None:
            raise ValueError(f'Invalid DatasetID string: {id_str}')
        path = m.group('path').replace('-', '/')
        subset = m.group('subset') or None
        return DatasetID(path=path, subset=subset)


@storage.Stringifiable.make_stringifiable()
class LLMType(storage.Stringifiable, enum.StrEnum, metaclass=EnumABCMeta):
    GIGACHAT = enum.auto()

    def __str__(self) -> str:
        return self.value

    def from_str(s: str) -> 'LLMType':
        for member in LLMType:
            if member.value == s:
                return member
        raise ValueError(f'Invalid LLMType string: {s}')


@storage.Storable.make_storable(storage.StorableType.LLM_LABELS)
@dataclasses.dataclass
class LLMLabels(storage.Storable):  # TODO: consider pool mapping
    dataset_id: DatasetID
    llm: LLMType
    labels: npt.NDArray[np.int64]

    def get_id(self) -> storage.ID:
        return f'{self.dataset_id}__{self.llm}#{self._get_salt(self.dataset_id, self.llm)}'

    @staticmethod
    def _get_salt(dataset_id: DatasetID, llm: LLMType) -> storage.Hash:
        return storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, (dataset_id, llm))))

    def as_storable(self) -> storage.StorableBundle:
        salt = self._get_salt(self.dataset_id, self.llm)
        array_id = f'{salt}_1#{hash((salt, 1))}'
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        'llm': str(self.llm),
                        'labels': array_id,
                    },
                    type=storage.StorableType.LLM_LABELS,
                ),
                array_id: storage.StorableEntry.from_npy(self.labels),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'LLMLabels':
        if entry.id in data and storable_type != storage.StorableType.LLM_LABELS:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = LLMLabels(
            dataset_id=DatasetID.from_str(payload['dataset_id']),
            llm=LLMType.from_str(payload['llm']),
            labels=data.retrieve(payload['labels']).payload['array'],
        )
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
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'Pool':
        if entry.id in data and storable_type != storage.StorableType.POOL:
            return data.retrieve(entry.id)
        payload = entry.payload
        pool = Pool(
            indices=data.retrieve(payload['indices']),
            dataset=data.retrieve(payload['dataset_id']).lazy_train,
        )
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


@storage.Storable.make_storable(storage.StorableType.DATASET)
class CompleteDataset(storage.Storable):  # TODO: Consider dumping on disk
    def __init__(
        self, id: DatasetID, text_field: str, label_field: str, database: 'DataDatabase', train_len: None | int = None
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

    @property
    def annotations(self) -> dict[LLMType, LLMLabels]:  # TODO: pass storage object
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
            LLMType.from_str(llm_str): LLMLabels.load(self.__root_dir, label_file)
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
        if self.id.subset is None:
            self.__dataset = datasets.load_dataset(self.id.path)
        else:
            self.__dataset = datasets.load_dataset(self.id.path, self.id.subset)
        self.__dataset = self.__standardize_dataset(self.__dataset)
        return self.__dataset

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
                    },
                    type=storage.StorableType.DATASET,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'CompleteDataset':
        if entry.id in data and storable_type != storage.StorableType.DATASET:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = CompleteDataset(
            id=DatasetID.from_str(payload['dataset']),
            text_field=payload['text_field'],
            label_field=payload['label_field'],
            database=data,
            train_len=payload['train_len'],
        )
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
                    },
                    type=storage.StorableType.DATASETS,
                    id=self.get_id(),
                ),
                **dict(item for d in self.datasets.values() for item in d.as_storable().entries.items()),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: 'DataDatabase', storable_type: storage.StorableType | None = None
    ) -> 'Datasets':
        if entry.id in data and storable_type != storage.StorableType.DATASETS:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = Datasets(data.retrieve(dataset_entry_id) for dataset_entry_id in payload['datasets'].values())
        data.encache(obj)
        return obj


class DataDatabase:
    BACKUPS_COUNT = 5

    def __init__(self, root_dir: pathlib.Path, local: bool = True):  # TODO  implement remote
        self.objects_store: dict[storage.ID, storage.Storable] = {}
        self.stored_index: dict[storage.ID, storage.StoredEntry] = {}
        self.root_dir = root_dir
        self.local = local
        self.__connected = False
        self.__datasets: Datasets = None
        self.__experiments: experiments.Experiments = None

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
            print(
                f'[WARNING]: {'Experiment' if isinstance(obj, experiments.Experiment) else 'ExperimentHistory'} {obj} with id {id} deleted from objects, but present in stored_index'
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
                f"Incorrect parameter mix: {inconsistent_ok = }, {inconsistent_on_disk_ok = }, {inconsistent_in_cache_ok = }. First is incompatible with 2 other"
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
                        print(
                            f'[WARNING]: for object with id {id} stored entry on disk {loaded.obj} differs from entry being stored {obj}. Trying to merge...'
                        )
                        try:
                            merged = self.merge_storables(obj, loaded.obj)
                        except ValueError:
                            print(
                                f'[WARNING]: Failed to merge {id}, preferring entry being stored {obj}, leaving disk one unchanged'
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
                    print(
                        f"[WARNING]: Can't merge storables {encached} and {was} with id {obj.get_id()}! Preferring encached {encached} for object {obj}"
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
                print(
                    f'[WARNING]: object being stored {obj} differs from encached object {encached}. Trying to merge...'
                )
                try:
                    merged = self.merge_storables(encached, obj)
                except ValueError:
                    print(
                        f'[WARNING]: Failed to merge {obj.id}, preferring the one just stored {obj}, differing from {encached}'
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
                print(
                    f'[WARNING]: entry being stored fast {bundle.entries[obj_id]} differs from entry already stored! Trying to merge...'
                )
                try:
                    merged = self.merge_storables(bundle.entries[obj_id], self.stored_index[obj_id].obj)
                except ValueError:
                    print(
                        f'[WARNING]: Failed to merge, preferring entry being stored {bundle.entries[obj_id]} over {self.stored_index[obj_id]}'
                    )
                    merged = self.__try_merge_storables(bundle.entries[obj_id], self.stored_index[obj_id].obj)
                self.add_to_store_index(merged, force=True)
                self.__store_entry(bundle.entries[obj_id], obj_format)
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
            case storage.StorableType.ARRAY:
                return pathlib.Path('bin')
            case storage.StorableType.POOL | storage.StorableType.SEEDED_INDICES:
                return pathlib.Path('markup')
            case storage.StorableType.EXPERIMENT_HISTORY:
                return pathlib.Path('experiments', 'histories')
            case storage.StorableType.DATASET:
                return pathlib.Path('datasets')
            case storage.StorableType.DATASETS | storage.StorableType.EXPERIMENTS:
                return pathlib.Path()
            case storage.StorableType.LLM_LABELS:
                assert False  # TODO: implement label storing
            case storage.StorableType.EXPERIMENT:
                return pathlib.Path('experiments')
            case _:
                assert False, 'unreachable'

    def __pre_store_entry(self, obj: 'storage.StorableEntry', where: pathlib.Path):
        where.parent.mkdir(parents=True, exist_ok=True)

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
                print(
                    f'[WARNING]: during dump for id {encached.id} inconsistent storable {self.stored_index[obj.get_id()].obj} and encached object {encached}! Preferring encached'
                )
                self.add_to_store_index(obj.as_storable(), force=True)

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
                            print(
                                f'[ERROR]: Failed to recollect object with id {obj_id}. Skipping {path}. On disk: {loaded.obj}, in RAM: {stored_entry.obj}.'
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
                            print(
                                f'[ERROR]: Failed to recollect object with id {obj_id}. Skipping {path}. On disk: {loaded.obj}, in RAM: {stored_entry.obj}.'
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
            print(f'[WARNING]: deleting file {file}')
            file.unlink()

        def warn_unindexed(folder: pathlib.Path, file: pathlib.Path):
            stored_type = storage.Format.type_from_format_name(file.name)
            if not stored_type.is_groupable():
                return
            stored_id = storage.Format.id_from_format_name(file.name)
            if stored_id not in self.stored_index:
                print(f'[WARNING]: unindexed file: {file} for id {stored_id} with type {stored_type}')

        self.__walk_local_storage(warn_unindexed)

    def try_restore(self):
        def process_file(folder: pathlib.Path, file: pathlib.Path):
            stored_id = storage.Format.id_from_format_name(file.name)
            loaded = self.__load_from_disk_by_path(stored_id, self.root_dir / storage.DATA_DIR / file)
            if stored_id not in self.stored_index and stored_id not in self.objects_store:
                self.add_to_store_index(loaded.obj)
            elif stored_id not in self.stored_index and stored_id in self.objects_store:
                print(f'[WARNING]: unindexed stored file: {file} that was recreated: {stored_id}')
                was = self.objects_store[stored_id]
                bundle = was.as_storable()
                entry = bundle.entries[bundle.main]
                merged = DataDatabase.merge_storables(entry, loaded.obj)
                self.destroy(merged.id, inconsistent_ok=True)
                self.add_to_store_index(merged)
            elif stored_id not in self.objects_store:
                if self.storables_differ(self.stored_index[stored_id].obj, loaded.obj):
                    merged = DataDatabase.merge_storables(loaded.obj, self.stored_index[stored_id].obj)
                    self.add_to_store_index(merged, force=True)
                else:
                    print(f'[WARNING]: Consistent stored on disk file: {file}. Marking as stored on disk')
                    self.stored_index[stored_id].stored = True
                    self.stored_index[stored_id].format = storage.Format.format_from_format_name(file.name)
            else:
                bundle = self.objects_store[stored_id].as_storable()
                entry = bundle.entries[bundle.main]
                assert not self.storables_differ(self.stored_index[stored_id].obj, entry)
                if self.storables_differ(self.stored_index[stored_id].obj, loaded.obj):
                    try:
                        merged = DataDatabase.merge_storables(loaded.obj, self.stored_index[stored_id].obj)
                    except ValueError as e:
                        print(
                            f'[WARNING]: `{e}`. Preferring live object, merging into it. Live: {self.stored_index[stored_id].obj}, loaded: {loaded.obj}',
                            end='',
                        )
                        merged = DataDatabase.__try_merge_storables(self.stored_index[stored_id].obj, loaded.obj)
                        print(f', merged: {merged.payload}')
                    self.destroy(merged.id, inconsistent_ok=True)
                    self.add_to_store_index(merged, force=True)
                    if not self.stored_index[stored_id].stored_on_disk:
                        # print(
                        #     f"[WARNING]: Merged id {loaded.obj.id} into memory, but didn't change file {file}, because object in database was not stored on disk. Merged: {merged}, loaded: {loaded.obj}"
                        # )
                        self.__store_entry(
                            self.stored_index[stored_id].obj, self.stored_index[stored_id].format or storage.Format.JSON
                        )
                else:
                    print(f'[WARNING]: Consistent stored on disk file: {file}. Marking as stored on disk')
                    self.stored_index[stored_id].stored = True
                    self.stored_index[stored_id].format = storage.Format.format_from_format_name(file.name)

        self.__walk_local_storage(process_file)

    def __walk_local_storage(self, callback: Callable[[pathlib.Path, pathlib.Path], None]):
        data_folder = self.root_dir / storage.DATA_DIR
        for folder in (
            data_folder / 'datasets',
            data_folder / 'experiments' / 'histories',
            data_folder / 'experiments',
            data_folder / 'markup',
            data_folder / 'bin',
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

    def get_dataset(self, obj: DatasetID, text_field: str = 'text', label_field: str = 'label') -> CompleteDataset:
        if obj in self.datasets:
            return self.datasets.datasets[obj]
        dataset = CompleteDataset(
            id=obj,
            text_field=text_field,
            label_field=label_field,
            database=self,
        )
        self.encache(dataset)

        return dataset

    def get_experiment(self, obj: 'experiments.Experiment') -> 'experiments.Experiment':
        if obj.get_id() in self.stored_index and obj.get_id() not in self.objects_store:
            raise RuntimeError(
                f'[ERROR]: Inconsistent stored_index and objects_store for Experiment; id: {obj.get_id()}, experiment: {self.stored_index[obj.get_id()]}'
            )
            # exp = experiments.Experiment.from_storable(self.stored_index[obj.get_id()].obj, self)
            # if exp not in self.experiments:
            #     self.experiments.add(exp)
            # return exp
        if obj in self.experiments:
            return self.experiments[obj][0]
        self.encache(obj)
        return obj

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
                    'path': str(  # TODO: rename everywhere and check `rel_path`
                        self.__get_rel_directory(entry.obj.type) / entry.format.format_name(obj_id, entry.obj.type)
                    ),
                    'format': entry.format.value,
                    # TODO: Add 'type': entry.obj.type.value,
                    # TODO: consider remote
                }
                for obj_id, entry in self.stored_index.items()
                if entry.stored_on_disk
                or (
                    not entry.obj.type.is_groupable()
                    and (
                        print(
                            f'[WARNING]: ungrouppable entry {entry.obj.id} is not stored on disk after pre dump during storables list generation! Entry contents: {entry}'
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
        return cls.load(f'{cls.get_config_name()}.json', root_dir, local=local)

    @classmethod
    def load(cls, config_path: pathlib.Path, root_dir: pathlib.Path, *, local: bool = True) -> 'DataDatabase':
        if not local:
            assert False, 'TODO: implement remote'
        if not config_path.exists():
            raise FileNotFoundError(f'Config file {config_path} does not exist')
        with config_path.open('r') as dbf:
            config = json.load(dbf)
        db = cls(root_dir, local)
        db.__load_config(config)
        return db

    def __load_config(self, config: dict):
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
        dead_ids = []
        for obj_id in config['refs']:
            try:
                self.stored_index[obj_id] = self.__load_from_disk(
                    config, obj_id
                )  # TODO: consider adding refs not to load everything on start
            except FileNotFoundError:
                print(f'[WARNING]: file for stored object with id {obj_id} not found, considering it dead id')
                dead_ids.append(obj_id)

        self.__recursively_clean_dead_ids(dead_ids)

        assert not self.objects_store
        assert not tuple(self.experiments)
        assert not tuple(self.datasets)

        for obj_id in itertools.chain(config['datasets'], config['experiments']):
            if obj_id not in self.objects_store and obj_id in self.stored_index:
                self.__recreate(self.stored_index[obj_id].obj, inconsistent_after_encache_ok=True)

    def __load_from_disk(self, config: dict, obj_id: 'storage.ID') -> 'storage.StoredEntry':
        if not self.local:
            assert False, 'TODO: implement remote'

        if obj_id not in config['refs']:
            raise KeyError(f'Object with id {obj_id} not found in config')
        content = config['refs'][obj_id]
        obj_format = storage.Format(content['format'])
        path: pathlib.Path = self.root_dir / storage.DATA_DIR / content['path']
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

    def __recursively_clean_dead_ids(self, dead_ids: deque[int]):
        to_be_died = set(dead_ids)
        while dead_ids:
            dead_id = dead_ids.popleft()

            tmp = dict(self.stored_index)
            for stored_id, entry in tmp.items():
                if self.stored_index[stored_id].obj.type in (
                    storage.StorableType.EXPERIMENTS,
                    storage.StorableType.DATASETS,
                    storage.StorableType.LLM_LABELS,
                ):  # Assumes everything is already loaded into stored_index
                    continue
                if dead_id in entry.obj.get_references():
                    print(f'[WARNING]: incomplete object {stored_id} removing from stored index. Content: {entry.obj}')
                    del self.stored_index[stored_id]
                    if stored_id not in to_be_died:
                        dead_ids.append(stored_id)
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
        return obj1.payload != obj2.payload

    @staticmethod
    def merge_storables(obj1: 'storage.StorableEntry', obj2: 'storage.StorableEntry') -> 'storage.StorableEntry':
        res1 = DataDatabase.__try_merge_storables(obj1, obj2)
        res2 = DataDatabase.__try_merge_storables(obj2, obj1)
        if res1.payload != res2.payload:
            raise ValueError(
                f'[ERROR]: Inconsistent merging! ids: {res1.id}, {res2.id}, type: {res1.type}, merge results: {res1.payload}, {res2.payload}'
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
        raise ValueError(f'Unknown types for merge: {main.type}')

    def __enter__(self):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError('Remote databases are not implemented yet')

        config_path = self.root_dir / storage.DATA_DIR / f'{self.get_config_name()}.json'
        if config_path.exists():
            tmp = DataDatabase.load(config_path, self.root_dir, local=self.local)
            self.objects_store = tmp.objects_store
            self.stored_index = tmp.stored_index
            self.__datasets = tmp.__datasets
            self.__experiments = tmp.__experiments
            tmp.objects_store = {}
            tmp.stored_index = {}
            tmp.__datasets = None
            tmp.__experiments = None
        self.__connected = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError('Remote databases are not implemented yet')
        self.dump()
        self.__connected = False
