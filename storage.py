import numpy as np
import numpy.typing as npt

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import Self
import hashlib
import pathlib
import contextlib
import json

DATA_DIR = pathlib.Path("data")

ID = str
Hash = str

class StorableType(enum.Enum): 
    ARRAY = enum.auto()
    LLM_LABELS = enum.auto()
    EXPERIMENT_HISTORY = enum.auto()
    EXPERIMENT = enum.auto()
    EXPERIMENTS = enum.auto()
    SEEDED_INDICES = enum.auto()
    POOL = enum.auto()
    DATASET = enum.auto()
    DATASETS = enum.auto()
    
    @classmethod
    def str_mappings(cls) -> dict[Self, str]:
        return {
            cls.ARRAY: "array",
            cls.LLM_LABELS: "llm_labels",
            cls.EXPERIMENT_HISTORY: "experiment_history",
            cls.EXPERIMENT: "experiment",
            cls.EXPERIMENTS: "experiments",
            cls.SEEDED_INDICES: "seeded_indices",
            cls.POOL: "pool",
            cls.DATASET: "dataset",
            cls.DATASETS: "datasets",
    }
    
    @classmethod
    def reverse_str_mappings(cls) -> dict[str, Self]:
        return {v: k for k, v in cls.str_mappings().items()}


    
@dataclasses.dataclass(slots=True, frozen=True)
class StorableEntry:
    payload: dict
    type: StorableType
    id: ID 

    @staticmethod
    def from_npy(arr: npt.NDArray) -> Self:
        return StorableEntry(payload={"array": arr}, type=StorableType.ARRAY)
    
    


@dataclasses.dataclass(slots=True)
class StorableBundle:
    main: str
    entries: dict[ID, StorableEntry]


class Storable(ABC): #TODO: from storable
    @abstractmethod
    def as_storable(self, *args) -> StorableBundle:
        pass

    @abstractmethod
    def get_id(self, *args) -> ID:
        pass

    @staticmethod
    @abstractmethod
    def _get_salt(*args) -> str:
        pass
    
    @staticmethod
    def hash_str(s: str) -> Hash:
        return hashlib.sha1(s.encode()).hexdigest()[:8]
    
    @staticmethod
    def combine_hashes(*hashes : Hash)-> Hash:
        return Storable.hash_str("".join(hashes))
    
class Stringifiable(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def from_str(s: str) -> Self:
        pass

class Format(enum.Enum):
    JSON = enum.auto()
    NPZ = enum.auto()
    
    def format_name(self, id: ID, entry_type: StorableType) -> str:
        match self:
            case Format.JSON:
                match entry_type:
                    case StorableType.ARRAY:
                        raise ValueError("ARRAY type cannot be stored in JSON format")
                    case StorableType.POOL:
                        return f"{id}_pool.json"
                    case StorableType.SEEDED_INDICES:
                        return f"{id}_seeded_indices.json"
                    case StorableType.LLM_LABELS:
                        assert False # TODO: implement label storing
                    case StorableType.DATASET:
                        return f"{id}_dataset.json"
                    case StorableType.DATASETS:
                        return f"{id}_datasets.json"
                    case StorableType.EXPERIMENT:
                        return f"{id}_experiment.json"
                    case StorableType.EXPERIMENTS:
                        return f"{id}_experiments.json"
                    case StorableType.EXPERIMENT_HISTORY:
                        return f"{id}_hist.json"
                    case _:
                        raise NotImplementedError(f"JSON format not implemented for {entry_type} type")
            case Format.NPZ:
                match entry_type:
                    case StorableType.ARRAY:
                        return f"{id}.npz"
                    case _:
                        raise ValueError(f"{entry_type} type cannot be stored in NPZ format")

@dataclasses.dataclass(slots=True)
class CacheMetadata:
    stored: bool = False
    format: Format | None = None

    @property
    def grouped(self) -> bool:
        return self.format is None

@dataclasses.dataclass(slots=True)
class CacheEntry:
    obj: StorableEntry
    meta: CacheMetadata
    
class Formatter:
    @staticmethod
    def dump(obj: StorableEntry, format: Format, filepath: pathlib.Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        match format:
            case Format.NPZ:
                np.savez_compressed(filepath, obj.payload["array"])
            case Format.JSON:
                json.dump({
                    'payload': obj.payload,
                    'type': StorableType.str_mappings()[obj.type],
                }, filepath.open("w"))
