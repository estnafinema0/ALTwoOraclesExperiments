from utils import EnumABCMeta, open_subbuild
import storage
import experiments
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text.integrations.transformers.datasets import TransformersDataset
import datasets
import numpy as np
import numpy.typing as npt

import itertools
import hashlib
import re
import dataclasses
import pathlib
import functools
import enum
import json
import uuid
import contextlib
from typing import Iterable, Self, Protocol


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
        return f"{seed}_{size}_{dataset_size}#{SeededIndices._get_salt(size, seed, dataset_size)}"

    def as_storable(self) -> storage.StorableBundle:
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        "seed": self.seed,
                        "size": self.size,
                        "dataset_size": self.dataset_size,
                    },
                    type=storage.StorableType.SEEDED_INDICES,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def from_storable(
        entry: "storage.StorableEntry", data: "DataDatabase", storable_type: storage.StorableType | None = None
    ) -> "SeededIndices":
        if entry.id in data and storable_type != storage.StorableType.SEEDED_INDICES:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = SeededIndices(size=payload["size"], seed=payload["seed"], dataset_size=payload["dataset_size"])
        data.encache(obj)
        return obj


@storage.Stringifiable.make_stringifiable()
@dataclasses.dataclass(frozen=True, eq=True, slots=True)
class DatasetID(storage.Stringifiable):
    path: str
    subset: str | None = None

    STR_PATTERN = re.compile(r"(?P<path>[a-zA-Z0-9_]+(-[a-zA-Z0-9_]+)*)_(?P<subset>([a-zA-Z0-9_]+)?)")

    def __str__(self) -> str:
        return f"{self.path.replace('/', '-')}_{[self.subset, ''][self.subset is None]}"

    @staticmethod
    def from_str(id_str: str) -> "DatasetID":
        m = re.fullmatch(DatasetID.STR_PATTERN, id_str)
        if m is None:
            raise ValueError(f"Invalid DatasetID string: {id_str}")
        path = m.group("path").replace("-", "/")
        subset = m.group("subset") or None
        return DatasetID(path=path, subset=subset)


@storage.Stringifiable.make_stringifiable()
class LLMType(storage.Stringifiable, enum.StrEnum, metaclass=EnumABCMeta):
    GIGACHAT = enum.auto()

    def __str__(self) -> str:
        return self.value

    def from_str(s: str) -> "LLMType":
        for member in LLMType:
            if member.value == s:
                return member
        raise ValueError(f"Invalid LLMType string: {s}")


@storage.Storable.make_storable(storage.StorableType.LLM_LABELS)
@dataclasses.dataclass
class LLMLabels(storage.Storable):  # TODO: consider pool mapping
    dataset_id: DatasetID
    llm: LLMType
    labels: npt.NDArray[np.int64]

    def get_id(self) -> storage.ID:
        return f"{self.dataset_id}__{self.llm}#{self._get_salt(self.dataset_id, self.llm)}"

    @staticmethod
    def _get_salt(dataset_id: DatasetID, llm: LLMType) -> storage.Hash:
        return storage.Storable.combine_hashes(*map(storage.Storable.hash_str, map(str, (dataset_id, llm))))

    def as_storable(self) -> storage.StorableBundle:
        salt = self._get_salt(self.dataset_id, self.llm)
        array_id = f"{salt}_1#{hash((salt, 1))}"
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        "llm": str(self.llm),
                        "labels": array_id,
                    },
                    type=storage.StorableType.LLM_LABELS,
                ),
                array_id: storage.StorableEntry.from_npy(self.labels),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: "DataDatabase", storable_type: storage.StorableType | None = None
    ) -> "LLMLabels":
        if entry.id in data and storable_type != storage.StorableType.LLM_LABELS:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = LLMLabels(
            dataset_id=DatasetID.from_str(payload["dataset_id"]),
            llm=LLMType.from_str(payload["llm"]),
            labels=data.retrieve(payload["labels"]).payload["array"],
        )
        data.encache(obj)
        return obj

    def __eq__(self, value):
        if not isinstance(value, LLMLabels):
            return False
        return (
            self.llm == value.llm and np.array_equal(self.labels, value.labels) and self.dataset_id == value.dataset_id
        )


class Dataset(Protocol):
    def __init__(self, dataset: datasets.Dataset, base: "CompleteDataset"):
        self.dataset = dataset
        self.base = base
        self.annotations = ...
        self.__y = None
        self.__x = None

    @property
    def y(self) -> npt.NDArray[np.int64]:
        if self.__y is None:
            self.__y = np.array(self.dataset["label"], dtype=np.int64)
        return self.__y

    @property
    def x(self) -> npt.NDArray[np.str_]:
        if self.__x is None:
            self.__x = np.array(self.dataset["text"], dtype=np.str_)
        return self.__x

    @property
    def size(self) -> int:
        return len(self.dataset)

    def __len__(self) -> int:
        return self.size

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
    def base(self) -> "CompleteDataset":
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
        return f"{dataset_id}__{indices_id}#{Pool._get_salt(dataset_id, indices_id)}"

    def as_storable(self) -> storage.StorableBundle:
        indices_entry = self.indices.as_storable()
        dataset_entry = storage.StorableEntry(
            payload={
                "dataset_id": CompleteDataset.make_id(self.base.id),
                "indices": self.indices.get_id(),
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
        entry: storage.StorableEntry, data: "DataDatabase", storable_type: storage.StorableType | None = None
    ) -> "Pool":
        if entry.id in data and storable_type != storage.StorableType.POOL:
            return data.retrieve(entry.id)
        payload = entry.payload
        return Pool(
            indices=data.retrieve(payload["indices"]),
            dataset=data.retrieve(payload["dataset_id"]).train,
        )

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
        return f"Pool(dataset_id={self.base.id}, indices={self.indices.indices})"

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
    def __init__(self, id: DatasetID, text_field: str, label_field: str, database: "DataDatabase"):
        self.id = id
        self.text_field = text_field
        self.label_field = label_field
        self.__dataset = None
        self.__annotations = None
        self.__train = None
        self.__validation = None
        self.__database = database

    @property
    def annotations(self) -> dict[LLMType, LLMLabels]:  # TODO: pass storage object
        if self.__annotations is not None:
            return self.__annotations
        self.__annotations = {}
        dataset_file = self.__root_dir / storage.DATA_DIR / self.get_config_filename()
        if dataset_file.exists():
            with dataset_file.open("r") as df:
                config = json.load(df)
            self.__load_annotations(config)
        return self.__annotations

    def __load_annotations(self, config: dict):
        self.__annotations = {
            LLMType.from_str(llm_str): LLMLabels.load(self.__root_dir, label_file)
            for llm_str, label_file in config["annotations"].items()
        }

    @property
    def train(self) -> Dataset:
        if self.__train is None:
            self.__train = Dataset(self.internal["train"], self)
        return self.__train

    @property
    def validation(self) -> Dataset:
        if self.__validation is None:
            self.__validation = Dataset(self.internal.get("validation", self.internal["test"]), self)
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
        obj = Pool(indices, self.train)
        self.__database.encache(obj)
        return obj

    @staticmethod
    def _get_salt(dataset_id: DatasetID) -> storage.Hash:
        return storage.Storable.hash_str(str(dataset_id))

    def get_id(self) -> storage.ID:
        return self.make_id(self.id)

    @staticmethod
    def make_id(dataset_id: DatasetID) -> storage.ID:
        return f"{dataset_id}#{CompleteDataset._get_salt(dataset_id)}"

    def as_storable(self) -> storage.StorableBundle:  # TODO: annotations
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        "text_field": self.text_field,
                        "label_field": self.label_field,
                        "dataset": str(self.id),
                    },
                    type=storage.StorableType.DATASET,
                    id=self.get_id(),
                ),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: "DataDatabase", storable_type: storage.StorableType | None = None
    ) -> "CompleteDataset":
        if entry.id in data and storable_type != storage.StorableType.DATASET:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = CompleteDataset(
            id=DatasetID.from_str(payload["dataset"]),
            text_field=payload["text_field"],
            label_field=payload["label_field"],
            database=data,
        )
        data.encache(obj)
        return obj

    def __standardize_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        return dataset.map(
            lambda ex: {
                "text": ex[self.text_field],
                "label": ex[self.label_field],
            },
            remove_columns=dataset["train"].column_names,
        )


@storage.Storable.make_storable(storage.StorableType.DATASETS)
class Datasets(storage.Storable):
    def __init__(self, *datasets: CompleteDataset):
        self.datasets = {dataset.id: dataset for dataset in datasets}

    def __contains__(self, key: CompleteDataset | DatasetID) -> bool:
        if isinstance(key, CompleteDataset):
            key = key.id
        return key in self.datasets

    def add(self, dataset: CompleteDataset) -> None:
        if dataset.id not in self.datasets:
            self.datasets[dataset.id] = dataset
            return
        raise ValueError("Dataset with the same storage.ID already exists in the collection")

    def __getitem__(self, key: DatasetID) -> CompleteDataset:
        if key in self.datasets:
            return self.datasets[key]
        raise KeyError("Dataset not found in the collection")

    def __iter__(self) -> Iterable[CompleteDataset]:
        return iter(self.datasets.values())

    @staticmethod
    def _get_salt() -> storage.Hash:
        return storage.Storable.hash_str("datasets")

    def get_id(self) -> storage.ID:
        return self.make_id()

    @staticmethod
    def make_id() -> storage.ID:
        return f"datasets#{Datasets._get_salt()}"

    def as_storable(self) -> storage.StorableBundle:
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        "datasets": {
                            str(dataset_id): CompleteDataset.make_id(dataset.id)
                            for dataset_id, dataset in self.datasets.items()
                        },
                    },
                    type=storage.StorableType.DATASETS,
                ),
                **dict(item for d in self.datasets.values() for item in d.as_storable().entries.items()),
            },
        )

    @staticmethod
    def from_storable(
        entry: storage.StorableEntry, data: "DataDatabase", storable_type: storage.StorableType | None = None
    ) -> "Datasets":
        if entry.id in data and storable_type != storage.StorableType.DATASETS:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = Datasets(data.retrieve(dataset_entry_id) for dataset_entry_id in payload["datasets"].values())
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
    def experiments(self) -> "experiments.Experiments":
        if self.__experiments is None:
            self.__experiments = experiments.Experiments()
        return self.__experiments

    def retrieve(self, id: storage.ID) -> storage.Storable:
        if id in self.objects_store:
            return self.objects_store[id]
        if id in self.stored_index:
            obj = storage.Storable.restore(self.stored_index[id].obj, self, self.stored_index[id].obj.type)
            if id not in self.objects_store:
                self.encache(obj)
            if isinstance(obj, CompleteDataset) and obj.id not in self.datasets:
                self.datasets.add(obj)
            if isinstance(obj, experiments.Experiment) and obj not in self.experiments:
                self.experiments.add(obj)
            return obj
        raise KeyError(f"Object with id {id} not found in the database")

    def encache(self, obj: storage.Storable):
        if obj.get_id() in self.objects_store:
            raise ValueError(f"Object with id {obj.get_id()} already exists in the cache")
        self.objects_store[obj.get_id()] = obj
        if obj.get_id() not in self.stored_index:
            self.add_to_store_index(obj.as_storable())

        if isinstance(obj, CompleteDataset):
            if obj.id not in self.datasets:
                self.datasets.add(obj)
        elif isinstance(obj, experiments.Experiment):
            if obj not in self.experiments:
                self.experiments.add(obj)

    def add_to_store_index(self, obj: storage.StorableBundle | storage.StorableEntry, *, force: bool = False):
        if isinstance(obj, storage.StorableEntry):
            if force or obj.id not in self.stored_index:
                self.stored_index[obj.id] = storage.StoredEntry(obj, stored=False, format=None)
            return
        if isinstance(obj, storage.StorableBundle):
            self.__add_to_store_index_rec(obj.main, obj, force=force)
            return
        raise ValueError("Object must be either StorableEntry or StorableBundle")

    def __add_to_store_index_rec(self, key: storage.ID, bundle: storage.StorableBundle, *, force: bool = False):
        if key in self.stored_index and not force:
            return
        entry = bundle.entries[key]
        self.add_to_store_index(entry, force=force)
        for ref in entry.get_references():
            self.__add_to_store_index_rec(ref, bundle, force=force)

    def __contains__(self, obj: "storage.ID | experiments.Experiment | DatasetID") -> bool:
        if isinstance(obj, storage.ID):
            return obj in self.objects_store or obj in self.stored_index
        if isinstance(obj, experiments.Experiment):
            return obj in self.experiments
        elif isinstance(obj, DatasetID):
            return obj in self.datasets
        return NotImplemented

    def store_fast(self, obj: "storage.StorableBundle", format: storage.Format, obj_id: storage.ID | None = None):
        if obj_id is not None and obj_id in self.stored_index and not self.stored_index[obj_id].stored:
            return
        if obj_id is None:
            obj_id = obj.main
        self.__store_entry(obj.entries[obj_id], format)
        for ref in obj.entries[obj_id].get_references():
            self.store_fast(obj, format.switch_format(obj.entries[ref].type), ref)
        if obj_id not in self.stored_index:
            self.stored_index[obj_id] = storage.StoredEntry(obj.entries[obj_id], stored=True, format=format)
        else:
            self.stored_index[obj_id].stored = True
            self.stored_index[obj_id].format = format

    def __store_entry(self, obj: "storage.StorableEntry", format: storage.Format):
        where = self.root_dir / storage.DATA_DIR / self.__get_directory(obj.type) / format.format_name(obj.id, obj.type)
        self.__pre_store_entry(obj, where)
        storage.Formatter.dump(obj, format, where)
        self.__post_store_entry(obj, where)

    def __get_directory(self, entry_type: storage.StorableType) -> pathlib.Path:
        match entry_type:
            case storage.StorableType.ARRAY:
                return pathlib.Path("bin")
            case storage.StorableType.POOL | storage.StorableType.SEEDED_INDICES:
                return pathlib.Path("markup")
            case storage.StorableType.EXPERIMENT_HISTORY:
                return pathlib.Path("experiments", "histories")
            case storage.StorableType.DATASET:
                return pathlib.Path("datasets")
            case storage.StorableType.DATASETS | storage.StorableType.EXPERIMENTS:
                return pathlib.Path()
            case storage.StorableType.LLM_LABELS:
                assert False  # TODO: implement label storing
            case storage.StorableType.EXPERIMENT:
                return pathlib.Path("experiments")
            case _:
                assert False, "unreachable"

    def __pre_store_entry(self, obj: "storage.StorableEntry", where: pathlib.Path):
        where.parent.mkdir(parents=True, exist_ok=True)

    def __post_store_entry(self, obj: "storage.StorableEntry", where: pathlib.Path):
        pass

    def dump(self):
        path = self.root_dir / storage.DATA_DIR / f"{self.get_config_name()}.json"
        last_backup_id = self.__pre_dump(path)
        if not self.local:
            assert False, "TODO: implement remote"
        config = self.__generate_storables(last_backup_id + 1)
        for entry in self.stored_index.values():
            if not entry.stored and not entry.obj.type.is_groupable():
                format = storage.Format.JSON.switch_format(entry.obj.type)
                self.__store_entry(entry.obj, format)
                entry.stored = True
                entry.format = format
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as dbf:
            json.dump(config, dbf)
        self.__post_dump(last_backup_id + 1)

    def __pre_dump(self, path: pathlib.Path) -> int:
        if not self.local:
            assert False, "TODO: implement remote"
        backup_id = 0
        if path.exists():
            with path.open("r") as dbf:
                old_config = json.load(dbf)
            backup_id = old_config.get("backup_id", 1)
            path.rename(path.parent / f"{self.get_config_name()}_{backup_id + 1}.json")
        for obj in self.objects_store.values():
            self.add_to_store_index(obj.as_storable(), force=True)

        return backup_id

    def __post_dump(self, backup_id: int):
        if not self.local:
            assert False, "TODO: implement remote"
        # breakpoint()
        killable_old = self.root_dir / storage.DATA_DIR / f"{self.get_config_name()}_{backup_id - 4}.json"
        if killable_old.exists():
            killable_old.unlink()

    def recollect_stored(self): ...  # TODO

    def get_dataset(self, obj: DatasetID, text_field: str = "text", label_field: str = "label") -> CompleteDataset:
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

    def get_experiment(self, obj: "experiments.Experiment") -> "experiments.Experiment":
        if obj in self.experiments:
            return self.experiments[experiments.Experiments.ExperimentKey.from_experiment(obj)][0]
        self.encache(obj)
        return obj

    @staticmethod
    def get_config_name() -> str:
        return "DONT_EVER_TOUCH_THIS_FILE_database8=D"

    def __generate_storables(self, backup_id) -> dict:
        return {
            "backup_id": backup_id,
            "objects": {
                id: {
                    "type": entry.obj.type.value,
                    "payload": entry.obj.payload,
                }
                for id, entry in self.stored_index.items()
                if not entry.stored and entry.obj.type.is_groupable()
            },
            "refs": {
                id: {
                    "path": str(self.__get_directory(entry.obj.type) / entry.format.format_name(id, entry.obj.type)),
                    "format": entry.format.value,
                    # TODO: consider remote
                }
                for id, entry in self.stored_index.items()
                if entry.stored or not entry.obj.type.is_groupable()
            },
            "experiments": [exp.get_id() for exp in self.experiments],
            "datasets": [dataset.get_id() for dataset in self.datasets],
        }

    @classmethod
    def load(cls, config_path: pathlib.Path, root_dir: pathlib.Path, local: bool = True) -> "DataDatabase":
        if not local:
            assert False, "TODO: implement remote"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist")
        with config_path.open("r") as dbf:
            config = json.load(dbf)
        db = cls(root_dir, local)
        db.__load_config(config)
        return db

    def __load_config(self, config: dict):
        for id, obj in config["objects"].items():
            self.stored_index[id] = storage.StoredEntry(
                obj=storage.StorableEntry(
                    payload=obj["payload"],
                    type=storage.StorableType(obj["type"]),
                    id=id,
                ),
                stored=False,
                format=None,
            )
        for id in config["refs"]:
            self.__load_from_disk(config, id)  # TODO: consider adding refs not to load everything on start
        for id in config["experiments"]:
            if id not in self.objects_store:
                obj = storage.Storable.restore(self.stored_index[id].obj, self)
                if obj.get_id() not in self.objects_store:
                    self.encache(obj)
                if obj not in self.experiments:
                    self.experiments.add(obj)
        for id in config["datasets"]:
            if id not in self.objects_store:
                obj = storage.Storable.restore(self.stored_index[id].obj, self)
                if obj.get_id() not in self.objects_store:
                    self.encache(obj)
                if obj not in self.datasets:
                    self.datasets.add(obj)

    def __load_from_disk(self, config: dict, id: storage.ID):
        if not self.local:
            assert False, "TODO: implement remote"

        if id not in config["refs"]:
            raise KeyError(f"Object with id {id} not found in config")
        content = config["refs"][id]
        format = storage.Format(config["refs"][id]["format"])
        path: pathlib.Path = self.root_dir / storage.DATA_DIR / content["path"]
        entry = storage.Formatter.load(format, path)
        self.stored_index[id] = storage.StoredEntry(obj=entry, stored=True, format=format)

    # def connect(self) -> "DataDatabase":
    #     if self.__connected:
    #         raise RuntimeError("Database is already connected, cannot connect again")
    #     self.__connected = True
    #     if not self.local:  # TODO: implement remote
    #         raise NotImplementedError("Remote databases are not implemented yet")
    #     if (self.root_dir / storage.DATA_DIR / f"{DataDatabase.get_config_name()}.json").exists():
    #         self.load()
    #     return self

    def __enter__(self):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError("Remote databases are not implemented yet")

        path = self.root_dir / storage.DATA_DIR / f"{self.get_config_name()}.json"
        if path.exists():
            tmp = DataDatabase.load(path, self.root_dir, self.local)
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
            raise NotImplementedError("Remote databases are not implemented yet")
        self.dump()
        self.__connected = False
