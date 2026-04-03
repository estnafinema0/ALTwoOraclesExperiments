import database

import numpy as np
import numpy.typing as npt

import dataclasses
import itertools
import enum
from abc import ABC, abstractmethod
from typing import Self, Callable, Iterable
import hashlib
import pathlib
import weakref
import json

DATA_DIR = pathlib.Path('data')

ID = str
Hash = str


class StorableType(enum.StrEnum):
    ARRAY = enum.auto()
    LLM_LABELS = enum.auto()
    EXPERIMENT_HISTORY = enum.auto()
    EXPERIMENT = enum.auto()
    EXPERIMENTS = enum.auto()
    SEEDED_INDICES = enum.auto()
    POOL = enum.auto()
    DATASET = enum.auto()
    DATASETS = enum.auto()
    ARRAY_WRAPPER = enum.auto()
    LLM_INDEX_RECAP_EXAMPLES = enum.auto()
    DIVERSITY_BASED_K_MEANS_CLUSTERS = enum.auto()
    LLM_CLUSTER_EXAMPLES = enum.auto()

    def is_groupable(self) -> bool:
        return self not in (StorableType.ARRAY, StorableType.EXPERIMENTS, StorableType.DATASETS)


@dataclasses.dataclass(slots=True, frozen=True, eq=True)
class StorableEntry:
    payload: dict
    type: StorableType
    id: ID

    @staticmethod
    def from_npy(arr: npt.NDArray, id: ID) -> Self:
        return StorableEntry(payload={'array': arr}, type=StorableType.ARRAY, id=id)

    def get_references(self) -> set[ID]:
        return set(map(lambda x: x[0], self.get_references_from_pythonish(self.type, self.payload)))

    @staticmethod
    def get_references_from_pythonish(entry_type: StorableType, payload: dict) -> Iterable[tuple[ID, StorableType]]:
        match entry_type:
            case t if t in (
                StorableType.ARRAY,
                StorableType.ARRAY_WRAPPER,
                StorableType.DATASET,
                StorableType.SEEDED_INDICES,
                StorableType.EXPERIMENT_HISTORY,
            ):
                return set()
            case StorableType.LLM_LABELS:
                return [(payload['labels'], StorableType.ARRAY)]
            case StorableType.POOL:
                return {
                    (payload['dataset_id'], StorableType.DATASET),
                    (payload['indices'], StorableType.SEEDED_INDICES),
                }
            case StorableType.DATASETS:
                return set(zip(payload['datasets'].values(), itertools.repeat(StorableType.DATASET)))
            case StorableType.EXPERIMENT:
                return {(payload['dataset'], StorableType.DATASET), (payload['pool'], StorableType.POOL)} | set(
                    zip(payload['histories'].values(), itertools.repeat(StorableType.EXPERIMENT_HISTORY))
                )
            case StorableType.EXPERIMENTS:
                return set(zip(payload['experiments'], itertools.repeat(StorableType.EXPERIMENT)))
            case (
                StorableType.LLM_INDEX_RECAP_EXAMPLES
                | StorableType.LLM_CLUSTER_EXAMPLES
                | StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS
            ):
                return [(payload['subset'], StorableType.ARRAY_WRAPPER)]
            case _:
                assert False, 'unreachable'


@dataclasses.dataclass(slots=True, eq=True)
class StorableBundle:
    main: str
    entries: dict[ID, StorableEntry]

    def get_references(self) -> set[ID]:
        return self.entries[self.main].get_references()


# TODO: references
class Storable(ABC):  # TODO: from storable
    storable_reverse = weakref.WeakValueDictionary()
    storable_classes = weakref.WeakValueDictionary()

    @abstractmethod
    def as_storable(self, *args) -> StorableBundle:
        pass

    @staticmethod
    @abstractmethod
    def from_storable(
        entry: StorableEntry, data: 'database.DataDatabase', storable_type: StorableType | None = None
    ) -> Self:
        pass

    @classmethod
    def restore(
        cls, entry: StorableEntry, data: 'database.DataDatabase', storable_type: StorableType | None = None
    ) -> 'Storable':
        if entry.type not in cls.storable_reverse:
            raise ValueError(f'Unknown stored type, unregistered: {entry.type}')
        return cls.storable_reverse[entry.type](entry, data, storable_type=storable_type)

    @abstractmethod
    def get_id(self, *args) -> ID:
        pass

    @staticmethod
    @abstractmethod
    def make_id(*args) -> ID:
        pass

    @staticmethod
    @abstractmethod
    def _get_salt(*args) -> str:
        pass

    @staticmethod
    def hash_str(s: str) -> Hash:
        return hashlib.sha1(s.encode()).hexdigest()[:8]

    @staticmethod
    def combine_hashes(*hashes: Hash) -> Hash:
        return Storable.hash_str(''.join(hashes))

    @classmethod
    def register(cls, index: StorableType, storable: 'Storable'):
        assert index not in cls.storable_reverse
        cls.storable_reverse[index] = storable.from_storable
        cls.storable_classes[index] = storable

    @classmethod
    def storable_factory(
        cls, database: 'database.DataDatabase', storable_type: StorableType, *args, **kwargs
    ) -> 'Storable':
        if storable_type not in cls.storable_reverse:
            raise ValueError(f'Unkown stored type, unregistered: {storable_type}')
        obj = cls.storable_classes[storable_type](*args, **kwargs)
        obj_id = obj.get_id()
        if obj_id not in database:
            database.encache(obj)
            return database.objects_store[obj_id]
        return database.retrieve(obj_id)

    @classmethod
    def make_storable[T](cls, index: StorableType) -> 'Callable[[type[T]], type[T]]':
        def _store(storable: type[T]) -> type[T]:
            cls.register(index, storable)
            return storable

        return _store


class Stringifiable(ABC):
    stringifiebles: 'list[Stringifiable]' = []

    @abstractmethod
    def __str__(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def from_str(s: str, db: 'database.DataDatabase') -> Self:
        pass

    @staticmethod
    def restore(s: str, db: 'database.DataDatabase') -> 'Stringifiable':
        if s not in Stringifiable.stringifiebles:
            raise ValueError(f'Unkown stringifiable, unregistered: {s}')
        for cls in Stringifiable.stringifiebles:
            try:
                return cls.from_str(s, db)
            except ValueError:
                continue
        raise ValueError(f'Unkown stringifiable, no class could parse: {s}')

    @classmethod
    def make_stringifiable[T](cls) -> 'Callable[[type[T]], type[T]]':
        def _store(storable: type[T]) -> type[T]:
            cls.register(storable)
            return storable

        return _store

    @classmethod
    def register(cls, storable: 'Stringifiable'):
        if storable in cls.stringifiebles:
            raise ValueError(f'Stringifiable {storable} already registered')
        cls.stringifiebles.append(storable)


class Format(enum.Enum):
    JSON = enum.auto()
    NPZ = enum.auto()

    def format_name(self, id: ID, entry_type: StorableType) -> str:
        match self:
            case Format.JSON:
                match entry_type:
                    case StorableType.ARRAY:
                        raise ValueError('ARRAY type cannot be stored in JSON format')
                    case StorableType.ARRAY_WRAPPER:
                        return f'{id}_array_wrapper.json'
                    case StorableType.POOL:
                        return f'{id}_pool.json'
                    case StorableType.SEEDED_INDICES:
                        return f'{id}_seeded_indices.json'
                    case StorableType.LLM_LABELS:
                        assert False  # TODO: implement label storing
                    case StorableType.DATASET:
                        return f'{id}_dataset.json'
                    case StorableType.DATASETS:
                        return f'{id}_datasets.json'
                    case StorableType.EXPERIMENT:
                        return f'{id}_experiment.json'
                    case StorableType.EXPERIMENTS:
                        return f'{id}_experiments.json'
                    case StorableType.EXPERIMENT_HISTORY:
                        return f'{id}_hist.json'
                    case StorableType.LLM_INDEX_RECAP_EXAMPLES:
                        return f'{id}_llm_index_recap_examples.json'
                    case StorableType.LLM_CLUSTER_EXAMPLES:
                        return f'{id}_llm_cluster_examples.json'
                    case StorableType.DIVERSITY_BASED_K_MEANS_CLUSTERS:
                        return f'{id}_llm_diversity_based_k_means_clusters.json'
                    case _:
                        raise NotImplementedError(f'JSON format not implemented for {entry_type} type')
            case Format.NPZ:
                match entry_type:
                    case StorableType.ARRAY:
                        return f'{id}.npz'
                    case _:
                        raise ValueError(f'{entry_type} type cannot be stored in NPZ format')
            case _:
                assert False, 'unreachable'

    @staticmethod
    def format_from_format_name(filename: str) -> Self:
        if filename.endswith('.npz'):
            return Format.NPZ
        elif filename.endswith('.json'):
            return Format.JSON
        else:
            raise ValueError(f'Unknown format for file: {filename}')

    @staticmethod
    def type_from_format_name(filename: str) -> StorableType:
        COMMON_CONFIG_EXTENSION = ['.json']

        def is_of_format(filename: str, base_postfix: str) -> bool:
            return any(filename.endswith(f'{base_postfix}{extension}') for extension in COMMON_CONFIG_EXTENSION)

        if is_of_format(filename, '_pool'):
            return StorableType.POOL
        elif is_of_format(filename, '_seeded_indices'):
            return StorableType.SEEDED_INDICES
        elif is_of_format(filename, '_dataset'):
            return StorableType.DATASET
        elif is_of_format(filename, '_datasets'):
            return StorableType.DATASETS
        elif is_of_format(filename, '_experiment'):
            return StorableType.EXPERIMENT
        elif is_of_format(filename, '_experiments'):
            return StorableType.EXPERIMENTS
        elif is_of_format(filename, '_hist'):
            return StorableType.EXPERIMENT_HISTORY
        elif is_of_format(filename, '_llm_index_recap_examples'):
            return StorableType.LLM_INDEX_RECAP_EXAMPLES
        elif is_of_format(filename, '_array_wrapper'):
            return StorableType.ARRAY_WRAPPER
        elif filename.endswith('.npz'):
            return StorableType.ARRAY
        else:
            raise ValueError(f'Unknown format for file: {filename}')

    @staticmethod
    def id_from_format_name(filename: str) -> ID:
        match Format.type_from_format_name(filename):
            case (
                StorableType.POOL
                | StorableType.DATASET
                | StorableType.DATASETS
                | StorableType.EXPERIMENT
                | StorableType.EXPERIMENTS
                | StorableType.EXPERIMENT_HISTORY
            ):
                return filename.rsplit('_', 1)[0]
            case StorableType.SEEDED_INDICES | StorableType.ARRAY_WRAPPER:
                return filename.rsplit('_', 2)[0]
            case StorableType.LLM_INDEX_RECAP_EXAMPLES:
                return filename.rsplit('_', 4)[0]
            case StorableType.ARRAY:
                return filename.rsplit('.', 1)[0]
            case value:
                raise ValueError(f'Unknown format: {value}')

    def switch_format(self, new_entry_type: StorableType) -> 'Format':
        match self:
            case Format.JSON:
                match new_entry_type:
                    case StorableType.ARRAY:
                        return Format.NPZ
                    case (
                        StorableType.POOL
                        | StorableType.SEEDED_INDICES
                        | StorableType.LLM_LABELS
                        | StorableType.DATASET
                        | StorableType.DATASETS
                        | StorableType.EXPERIMENT
                        | StorableType.EXPERIMENTS
                        | StorableType.EXPERIMENT_HISTORY
                        | StorableType.LLM_INDEX_RECAP_EXAMPLES
                        | StorableType.ARRAY_WRAPPER
                    ):
                        return Format.JSON
                    case _:
                        raise NotImplementedError(f'JSON format not implemented for {new_entry_type} type')
            case Format.NPZ:
                raise ValueError(f'{new_entry_type} type cannot be stored in NPZ format')
            case _:
                assert False, 'unreachable'


@dataclasses.dataclass(slots=True)
class StoredEntry:
    obj: StorableEntry
    stored: bool = False
    format: Format | None = None

    @property
    def grouped(self) -> bool:
        return self.format is None

    @property
    def stored_on_disk(self) -> bool:
        return self.stored and not self.grouped


class Formatter:
    @staticmethod
    def dump(obj: StorableEntry, format: Format, filepath: pathlib.Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        match format:
            case Format.NPZ:
                np.savez_compressed(filepath, arr=obj.payload['array'])
            case Format.JSON:
                json.dump(
                    {
                        'payload': obj.payload,
                        'type': obj.type.value,
                        'id': obj.id,
                    },
                    filepath.open('w'),
                )
            case _:
                assert False, 'unreachable'

    @staticmethod
    def load(format: Format, filepath: pathlib.Path) -> StorableEntry:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        match format:
            case Format.NPZ:
                return StorableEntry(
                    payload={'array': np.load(filepath)['arr']},
                    type=StorableType.ARRAY,
                    id='',
                )
            case Format.JSON:
                with filepath.open('r') as f:
                    content = json.load(f)
                return StorableEntry(
                    payload=content['payload'],
                    type=StorableType(content['type']),
                    id=content['id'],
                )
            case _:
                assert False, 'unreachable'
