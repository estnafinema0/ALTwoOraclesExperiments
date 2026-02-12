from utils import EnumABCMeta, open_subbuild
import storage
import experiments
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text.integrations.transformers.datasets import TransformersDataset
import datasets
import numpy as np
import numpy.typing as npt

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

    # def dump(self, root_dir: pathlib.Path, external_id: str, filename: str | None = None):
    #     config = self.get_config(external_id)
    #     if config["hardcoded"]:
    #         np.save(root_dir / config["indices_file"], self.indices)

    #     with open_subbuild(
    #         root_dir / storage.DATA_DIR / (filename if filename is not None else self.get_config_filename(external_id))
    #     ).open("w") as cfg:
    #         json.dump(config, cfg)

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str) -> "Indices":
    #     with open_subbuild(root_dir / storage.DATA_DIR / filename).open("r") as cfg:
    #         config = json.load(cfg)
    #     if config["hardcoded"]:
    #         indices = np.load(root_dir / config["indices_file"])
    #         return Indices(indices=indices, repr=None)
    #     else:
    #         return Indices.from_seed(**config["repr"])


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
        return f"{self.seed}_{self.size}_{self.dataset_size}#{self._get_salt(self.size, self.seed, self.dataset_size)}"

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


class LLMType(storage.Stringifiable, enum.Enum, metaclass=EnumABCMeta):
    GIGACHAT = enum.auto()

    @classmethod
    def str_mappings(cls) -> dict[Self, str]:
        return {
            cls.GIGACHAT: "gigachat",
        }

    @classmethod
    def reverse_str_mappings(cls) -> dict[str, Self]:
        return {v: k for k, v in cls.str_mappings().items()}

    def __str__(self) -> str:
        return self.str_mappings()[self]

    @classmethod
    def from_str(cls: "LLMType", strategy_str: str) -> "LLMType":
        return cls.reverse_str_mappings()[strategy_str]


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

    # def dump(self, root_dir: pathlib.Path, external_id: str, filename: str | None = None):
    #     config = self.get_config(external_id)
    #     with (
    #         open_subbuild(
    #             root_dir / storage.DATA_DIR / (filename if filename is not None else self.get_config_filename(external_id))
    #         ).open("w") as cfg,
    #         open_subbuild(root_dir / config["labels"]).open("wb") as lf,
    #     ):
    #         json.dump(config, cfg)
    #         np.save(lf, self.labels)

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str) -> "LLMLabels":
    #     with (root_dir / storage.DATA_DIR / filename).open("r") as cfg:
    #         config = json.load(cfg)
    #     return LLMLabels(llm=LLMType.from_str(config["llm"]), labels=np.load(root_dir / config["labels"]))

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
            storage.Storable.hash_str(str(dataset_id)), storage.Storable.hash_str(indices_id)
        )

    def get_id(self) -> storage.ID:
        return f"{self.base.id}__{self.indices.get_id()}#{self._get_salt(self.base.id, self.indices.get_id())}"

    def as_storable(self) -> storage.StorableBundle:
        indices_entry = self.indices.as_storable()
        dataset_entry = storage.StorableEntry(
            payload={
                "dataset_id": str(self.base.id),
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
            },
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

    # def dump(self, root_dir: pathlib.Path, external_id: str, filename: str | None = None):
    #     config = self.get_config(external_id)
    #     with open_subbuild(
    #         root_dir / storage.DATA_DIR / (filename if filename is not None else self.get_config_filename(external_id))
    #     ).open("w") as cfg:
    #         json.dump(config, cfg)
    #     self.indices.dump(root_dir, external_id)
    #     for key, file in config["pool_files"].items():
    #         self.pool.dataset[key].save_to_disk(root_dir / file)
    #     for llm, file in config["annotations"].items():
    #         self.annotations[LLMType.from_str(llm)].dump(root_dir, external_id)

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str, dataset: CompleteDataset | None = None) -> "Pool":
    #     config_path = filename if filename.is_absolute() else root_dir / filename
    #     with config_path.open("r") as cfg:
    #         config = json.load(cfg)
    #     indices_file = pathlib.Path(config["indices_file"])

    #     if indices_file.parents[-2] == storage.DATA_DIR:
    #         indices_file = indices_file.relative_to(storage.DATA_DIR)
    #     indices = Indices.load(root_dir, indices_file)
    #     pool_data = datasets.DatasetDict()
    #     for key, file in config["pool_files"].items():
    #         pool_data[key] = datasets.Dataset.load_from_disk(root_dir / file)
    #     pool = Pool(indices=indices, dataset=dataset)
    #     pool.__pool = pool_data
    #     pool.__annotations = {
    #         LLMType.from_str(llm_str): LLMLabels.load(
    #             root_dir,
    #             (
    #                 pathlib.Path(label_file).relative_to(storage.DATA_DIR)
    #                 if pathlib.Path(label_file).parents[-2] == storage.DATA_DIR
    #                 else label_file
    #             ),
    #         )
    #         for llm_str, label_file in config["annotations"].items()
    #     }
    #     return pool


class CompleteDataset(storage.Storable):
    def __init__(self, id: DatasetID, text_field: str, label_field: str):  # TODO: pass storage object
        self.id = id
        self.text_field = text_field
        self.label_field = label_field
        self.__dataset = None
        self.__annotations = None
        self.__train = None
        self.__validation = None
        self.__pool = None

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

    def create_pool(self, indices: Indices) -> Pool:
        if self.__pool is not None:
            print("[WARNING]: Pool already exists for this dataset, overwriting")
        self.__pool = Pool(indices=indices, dataset=self.train)
        return self.__pool

    @property
    def pool(self) -> Pool:
        if self.__pool is None:
            raise ValueError("Firstly create pool via Dataset.create_pool method")
        return self.__pool

    @staticmethod
    def _get_salt(dataset_id: DatasetID) -> storage.Hash:
        return storage.Storable.hash_str(str(dataset_id))

    def get_id(self) -> storage.ID:
        return f"{self.id}#{self._get_salt(self.id)}"

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

    def __standardize_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        return dataset.map(
            lambda ex: {
                "text": ex[self.text_field],
                "label": ex[self.label_field],
            },
            remove_columns=dataset["train"].column_names,
        )

    # def dump(self, root_dir: pathlib.Path, filename: str | None = None):
    #     config = self.get_config()
    #     with open_subbuild(root_dir, storage.DATA_DIR, filename if filename is not None else self.get_config_filename()).open(
    #         "w"
    #     ) as df:
    #         json.dump(config, df)
    #     for labels in self.annotations.values():
    #         labels.dump(root_dir, str(self.id))

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str) -> "CompleteDataset":
    #     with (root_dir / filename).open("r") as df:
    #         config = json.load(df)
    #     dataset = CompleteDataset(
    #         id=DatasetID.from_str(config["id"]),
    #         text_field=config["text_field"],
    #         label_field=config["label_field"],
    #     )
    #     dataset.__load_annotations(config)
    #     return dataset


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
        return f"datasets#{self._get_salt()}"

    def as_storable(self) -> storage.StorableBundle:
        return storage.StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): storage.StorableEntry(
                    payload={
                        "datasets": {str(dataset_id): str(dataset.id) for dataset_id, dataset in self.datasets.items()},
                    },
                    type=storage.StorableType.DATASETS,
                ),
                **dict(item for d in self.datasets.values() for item in d.as_storable().entries.items()),
            },
        )


class DataDatabase:
    BACKUPS_COUNT = 5

    def __init__(self, root_dir: pathlib.Path, local: bool = True):  # TODO  implement remote
        self.objects_store: dict[storage.ID, storage.CacheEntry] = {}
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

    def store_fast(self, obj: storage.StorableBundle, format: storage.Format, obj_id: storage.ID | None = None):
        if obj_id is not None and obj_id in self.objects_store:
            return

        if obj_id is None:
            obj_id = obj.main

        cache_entry = storage.CacheEntry(obj=obj.entries[obj_id], meta=storage.CacheMetadata(stored=False, format=None))
        self.objects_store[obj_id] = cache_entry
        self.__store_entry(obj, cache_entry.obj, format)
        self.objects_store[obj_id].meta.stored = True
        self.objects_store[obj_id].meta.format = format
        if obj.entries[obj_id].type == storage.StorableType.EXPERIMENT:
            ...  # TODO: from_storable
        if obj.entries[obj_id].type == storage.StorableType.EXPERIMENT_HISTORY:
            ...  # TODO: create experiment and add

    def __store_entry(self, bundle: storage.StorableBundle, entry: storage.StorableEntry, format: storage.Format):
        filepath = self.root_dir / storage.DATA_DIR
        match entry.type:
            case storage.StorableType.ARRAY:
                filepath = filepath / "bin"
                if format != storage.Format.NPZ:
                    raise ValueError("Only NPZ format is supported for arrays")
            case t if t in (storage.StorableType.POOL, storage.StorableType.SEEDED_INDICES):
                filepath = filepath / "markup"
            case storage.StorableType.EXPERIMENT_HISTORY:
                filepath = filepath / "experiments" / "hisories"
            case storage.StorableType.DATASET:
                filepath = filepath / "datasets"
            case t if t in (storage.StorableType.DATASETS, storage.StorableType.EXPERIMENTS):
                pass
            case storage.StorableType.LLM_LABELS:
                ...  # TODO: store labels
            case storage.StorableType.EXPERIMENT:
                filepath = filepath / "experiments"
            case storage.StorableType.EXPERIMENT_HISTORY:
                filepath = filepath / "experiments" / "histories"
        filepath /= format.format_name(entry.id, entry.type)
        match entry.type:
            case storage.StorableType.DATASET:
                ...  # TODO: implement move logic that is in 677 to 716
            case t if t in (
                storage.StorableType.POOL,
                storage.StorableType.SEEDED_INDICES,
                storage.StorableType.EXPERIMENT_HISTORY,
                storage.StorableType.ARRAY,
                storage.StorableType.EXPERIMENT,
                storage.StorableType.EXPERIMENT_HISTORY,
            ):
                storage.Formatter.dump(entry, format, filepath)

        match entry.type:
            case storage.StorableType.POOL:
                self.store_fast(bundle, format, entry.payload["indices"])
            case t if t in (
                storage.StorableType.SEEDED_INDICES,
                storage.StorableType.EXPERIMENT_HISTORY,
                storage.StorableType.DATASET,
                storage.StorableType.ARRAY,
            ):
                pass
            case storage.StorableType.DATASETS:
                for dataset_id in bundle.entries[bundle.main].payload["datasets"]:
                    self.store_fast(bundle, format, dataset_id)
            case storage.StorableType.EXPERIMENTS:
                for experiment_id in bundle.entries[bundle.main].payload["experiments"]:
                    self.store_fast(bundle, format, experiment_id)
            case storage.StorableType.EXPERIMENT:
                for history_id in bundle.entries[bundle.main].payload["histories"].values():
                    self.store_fast(bundle, format, history_id)
            case storage.StorableType.LLM_LABELS:
                ...  # TODO

    def recollect_stored(self): ...  # TODO

    def retrieve(self, experiment: "experiments.Experiment") -> "experiments.Experiment":
        if experiment in self.experiments:
            key = experiments.Experiments.ExperimentKey.from_experiment(experiment)
            return self.experiments[key][0]
        return experiment

    def __contains__(self, experiment: "experiments.Experiment | DatasetID") -> bool:
        if isinstance(experiment, experiments.Experiment):
            return experiment in self.experiments
        elif isinstance(experiment, DatasetID):
            return experiment in self.datasets

    def get(self, obj: DatasetID, text_field: str = "text", label_field: str = "label") -> CompleteDataset:
        if isinstance(obj, DatasetID):
            if obj in self.datasets:
                return self.datasets.datasets[obj]
            dataset = CompleteDataset(
                id=obj,
                text_field=text_field,
                label_field=label_field,
            )
            self.datasets.add(dataset)
            self.objects_store.update(
                {
                    id: storage.CacheEntry(
                        obj=entry, meta=storage.CacheMetadata(stored=False, format=storage.Format.JSON)
                    )
                    for id, entry in dataset.as_storable().entries.items()
                }
            )

            return dataset

    @staticmethod
    def get_config_name() -> str:
        return "DONT_EVER_TOUCH_THIS_FILE_database8=D"

    def generate_storables(self, backup_id) -> dict:
        return {
            "backup_id": backup_id,
            "backup_file": f"{self.get_config_name()}_{backup_id}.json",
            "objects": {
                id: {
                    "type": entry.obj.type.value,
                    "payload": entry.obj.payload,
                }
                for id, entry in self.objects_store.items()
                if not entry.meta.stored
            },
            "experiments": [
                id
                for id, entry in self.objects_store.items()
                if entry.obj.type == storage.StorableType.EXPERIMENT
            ],
            "databases": {
                id: entry.obj.payload['dataset']
                for id, entry in self.objects_store.items()
                if entry.obj.type == storage.StorableType.DATASET
            }
        }

    def connect(self) -> "DataDatabase":
        if self.__connected:
            raise RuntimeError("Database is already connected, cannot connect again")
        self.__connected = True
        if not self.local:  # TODO: implement remote
            raise NotImplementedError("Remote databases are not implemented yet")
        if (self.root_dir / storage.DATA_DIR / f"{DataDatabase.get_config_name()}.json").exists():
            self.load()
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        for id, entry in self.objects_store.items():
            if not entry.meta.stored:
                self.__store_entry(
                    storage.StorableBundle(main=id, entries={id: entry.obj}),
                    entry.obj,
                    storage.Format.JSON if entry.obj.type != storage.StorableType.ARRAY else storage.Format.NPZ,
                )
        # TODO: save datasets and experiments
        self.dump()
        self.__connected = False

    def dump(self): # TODO: remove old files
        path = self.root_dir / storage.DATA_DIR / f"{self.get_config_name()}.json"
        if not self.local:
            ...  # TODO: implement remote
        old_config = {}
        if path.exists():
            with path.open("r") as dbf:
                old_config = json.load(dbf)
            old_path = path.rename(
                self.root_dir / storage.DATA_DIR / f"{self.get_config_name()}_{old_config['backup_id'] + 1}.json"
            )
        last_backup_id = old_config.get("backup_id", 1)
        config = self.generate_storables(last_backup_id + 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as dbf:
            json.dump(config, dbf)

    def load(self):
        if not self.local:  # TODO: implement remote
            raise NotImplementedError("Remote databases are not implemented yet")
        db = DataDatabase(self.root_dir, self.local)
        config_path = self.root_dir / storage.DATA_DIR / f"{db.get_config_name()}.json"
        with config_path.open("r") as dbf:
            config = json.load(dbf)
        for id, entry in config["objects"].items():
            ...  # TODO: implement from_storable

    # @staticmethod
    # def __is_experiments_subset(cfg1: dict, cfg2: dict) -> bool:
    #     return cfg1["root_dir"] == cfg2["root_dir"] and set(cfg1["experiments"]).issubset(set(cfg2["experiments"]))

    # @staticmethod
    # def __is_datasets_subset(cfg1: dict, cfg2: dict, root_dir: pathlib.Path, datasets: Datasets) -> bool:
    #     if not set(cfg1["datasets"]).issubset(set(cfg2["datasets"])):
    #         return False
    #     for key in cfg1["datasets"]:
    #         ds1 = CompleteDataset.load(root_dir, cfg1["datasets"][key])
    #         ds2 = datasets[DatasetID.from_str(key)]
    #         annot1 = ds1.annotations
    #         annot2 = ds2.annotations
    #         if set(annot1) != set(annot2):
    #             return False

    #         if not all(annot1[llm] == annot2[llm] for llm in annot1):
    #             return False
    #     return True

    # def dump(self, root_dir: pathlib.Path, filename: str | None = None):
    #     config = self.get_config()
    #     if (self.root_dir / storage.DATA_DIR / self.get_config_filename()).exists():
    #         (self.root_dir / storage.DATA_DIR / self.get_config_filename()).rename(
    #             self.root_dir / storage.DATA_DIR / config["backup_file"]
    #         )
    #     if (
    #         self.root_dir
    #         / storage.DATA_DIR
    #         / f"{self.get_config_filename()[:-5]}_{(config['backup_id'] - self.BACKUPS_COUNT)}.json"
    #     ).exists():
    #         (
    #             self.root_dir
    #             / storage.DATA_DIR
    #             / f"{self.get_config_filename()[:-5]}_{(config['backup_id'] - self.BACKUPS_COUNT)}.json"
    #         ).unlink()

    #     with open_subbuild(
    #         root_dir / storage.DATA_DIR / (filename if filename is not None else self.get_config_filename())
    #     ).open("w") as dbf:
    #         json.dump(config, dbf)

    #     old_exp_path = root_dir / storage.DATA_DIR / self.EXPERIMENTS_FILE
    #     if old_exp_path.exists():
    #         with old_exp_path.open("r") as ef:
    #             old_experiments = json.load(ef)

    #     experiments = self.experiments.to_json()
    #     if old_exp_path.exists() and not DataDatabase.__is_experiments_subset(old_experiments, experiments):
    #         new_filename = f"{self.get_config_filename()[:-5]}_{uuid.uuid4()}.json"
    #         print(f"WARNING: Some experiments are missing, saved previous file in {new_filename}")
    #         (self.root_dir / storage.DATA_DIR / self.EXPERIMENTS_FILE).rename(self.root_dir / storage.DATA_DIR / new_filename)

    #     with open_subbuild(root_dir / storage.DATA_DIR / self.EXPERIMENTS_FILE).open("w") as ef:
    #         json.dump(self.experiments.to_json(), ef)

    #     old_datasets_path = root_dir / storage.DATA_DIR / self.DATASETS_FILE
    #     if old_datasets_path.exists():
    #         with old_datasets_path.open("r") as df:
    #             old_datasets = json.load(df)

    #     datasets_config = old_datasets if old_datasets_path.exists() else self.datasets.to_json()

    #     if old_datasets_path.exists() and not DataDatabase.__is_datasets_subset(
    #         old_datasets, self.datasets.to_json(), root_dir, self.datasets
    #     ):
    #         salt = f"{uuid.uuid4()}"
    #         new_filename = f"{self.get_config_filename()[:-5]}_{salt}.json"
    #         print(
    #             f"WARNING: Some datasets are missing or have different annotations, saved previous file in {new_filename}"
    #         )
    #         (self.root_dir / storage.DATA_DIR / self.DATASETS_FILE).rename(self.root_dir / storage.DATA_DIR / new_filename)
    #         for dataset_id in old_datasets["datasets"]:
    #             renamed = False
    #             ds1 = CompleteDataset.load(root_dir, old_datasets["datasets"][dataset_id])
    #             ds2 = self.datasets[DatasetID.from_str(dataset_id)]
    #             annot1 = ds1.annotations
    #             annot2 = ds2.annotations
    #             cfg1 = ds1.get_config()
    #             for llm in annot1:
    #                 if not (annot1[llm] == annot2[llm]):
    #                     print(
    #                         f"WARNING: Annotations for dataset {dataset_id} and LLM {llm} differ between old and new database"
    #                     )
    #                     filename = cfg1["annotations"][str(llm)]
    #                     new_annotations_filename = f"{filename[:-5]}_{salt}.json"
    #                     (self.root_dir / storage.DATA_DIR / filename).rename(
    #                         self.root_dir / storage.DATA_DIR / new_annotations_filename
    #                     )
    #                     if not renamed:
    #                         new_dataset_filename = f"{old_datasets['datasets'][dataset_id][:-5]}_{salt}.json"
    #                         (self.root_dir / storage.DATA_DIR / old_datasets["datasets"][dataset_id]).rename(
    #                             self.root_dir / storage.DATA_DIR / new_dataset_filename
    #                         )

    #                         renamed = True
    #                     cfg1["annotations"][str(llm)] = new_annotations_filename
    #             if renamed:
    #                 with open_subbuild(self.root_dir / storage.DATA_DIR / new_dataset_filename).open("w") as df:
    #                     json.dump(cfg1, df)
    #                 datasets_config["datasets"][dataset_id] = ds1.get_config_filename()

    #     with open_subbuild(root_dir / storage.DATA_DIR / self.DATASETS_FILE).open("w") as df:
    #         json.dump(datasets_config, df)
    #     if self.__enter:
    #         return
    #     for dataset in self.datasets:
    #         dataset.dump(root_dir, datasets_config["datasets"][str(dataset.id)])

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str) -> "DataDatabase":
    #     raise RuntimeError("Use DataDatabase(root_dir) instead of load")

    # def __enter__(self) -> "DataDatabase":
    #     self.__enter = True
    #     self.__exit_stack = contextlib.ExitStack()
    #     for dataset in self.datasets:

    #         self.__exit_stack.enter_context(dataset(self.root_dir))
    #     return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.__enter = False
    #     self.__exit_stack.close()
    #     self.dump(self.root_dir)
