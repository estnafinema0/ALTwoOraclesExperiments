from utils import Stringifiable, Dumpable, JSONifiable

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text.integrations.transformers.datasets import TransformersDataset
import datasets
import numpy as np
import numpy.typing as npt

import dataclasses
import enum
import os
import json

DATA_DIR = 'data'

class Indices(Dumpable):
    def __init__(self, indices: npt.NDArray[np.int64], repr: dict[str, any] | None):
        self.indices = indices
        self.repr = repr

    @property
    def size(self) -> int:
        return len(self.indices)
    
    @staticmethod
    def from_seed(*, size: int, seed: int, dataset_size: int) -> 'Indices':
        rng = np.random.default_rng(seed)
        indices = np.arange(dataset_size)
        rng.shuffle(indices)
        return Indices(indices=indices[:size], repr={"seed": seed, "size": size, "dataset_size": dataset_size})

    def get_config_filename(self, external_id: str) -> str:
        return f"{external_id}_indices.json"
    
    def get_config(self, external_id: str) -> dict:
        config = {
            'hardcoded': self.repr is None, 
        }
        if self.repr is not None:
            config['repr'] = self.repr
        else:
            indices_file =  os.path.join(DATA_DIR, f"{external_id}_indices.dat") 
            config['indices_file'] = indices_file
        return config

    def dump(self, root_dir: os.PathLike, external_id: str, filename: str | None = None):
        config = self.get_config(external_id)
        if config['hardcoded']:
            np.save(os.path.join(root_dir, config['indices_file']), self.indices)

        with open(os.path.join(root_dir, DATA_DIR, filename if filename is not None else self.get_config_filename(external_id)), 'w') as cfg:
            json.dump(config, cfg)

    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> 'Indices':
        with open(os.path.join(root_dir, DATA_DIR, filename), 'r') as cfg:
            config = json.load(cfg)
        if config['hardcoded']:
            indices = np.load(os.path.join(root_dir, config['indices_file']))
            return Indices(indices=indices, repr=None)
        else:
            return Indices.from_seed(**config['repr'])
        
    def __call__(self, root_dir: os.PathLike, filename: str) -> 'Indices':
        self.__filename = filename
        self.__root_dir = root_dir
        return self
    
    def __enter__(self) -> 'Indices':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__root_dir is None or self.__filename is None:
            print("Warning: cannot dump Indices on exit")
        else:
            self.dump(self.__root_dir, self.__filename)
@dataclasses.dataclass
class DatasetID(Stringifiable):
    path: str
    subset: str | None = None

    def __str__(self) -> str:
        if self.subset is None:
            return self.path
        return f"{self.path.replace('/', '_')} {self.subset}"
    
    @staticmethod
    def from_str(id_str: str) -> 'DatasetID':
        if not id_str.endswith('None'):
            parts = id_str.split(' ')
            path = parts[0].replace('_', '/')
            subset = parts[-1]
            return DatasetID(path=path, subset=subset)
        else:
            return DatasetID(path=id_str, subset=None)
    
class LLMType(enum.Enum, Stringifiable):
    GIGACHAT = enum.auto()

    STR_MAPPINGS = {
        GIGACHAT: "gigachat",
    }

    REVERSE_STR_MAPPINGS = {v: k for k, v in STR_MAPPINGS.items()}

    def __str__(self) -> str:
        return self.STR_MAPPINGS[self]
    
    @classmethod
    def from_str(cls: 'LLMType', strategy_str: str) -> 'LLMType':
        return cls.REVERSE_STR_MAPPINGS[strategy_str]


@dataclasses.dataclass
class LLMLabels(Dumpable):
    llm: LLMType
    labels: npt.NDArray[np.int64]
    
    @staticmethod
    def id_to_file_name(ID: str, llm: LLMType) -> str:
        return f"{ID}_{llm}"

    def get_config_filename(self, external_id: str):
        return f"{LLMLabels.id_to_file_name(external_id, self.llm)}.json"
    
    def get_config(self, external_id: str) -> dict:
        label_file =  os.path.join(DATA_DIR, f"{LLMLabels.id_to_file_name(external_id, self.llm)}.dat") 
        config = {
            'llm': str(self.llm), 
            'labels': label_file, 
        }
        return config
    
    def dump(self, root_dir: os.PathLike, external_id: str, filename: str | None = None):
        config = self.get_config(external_id)
        with (open(os.path.join(root_dir, DATA_DIR, filename if filename is not None else self.get_config_filename(external_id)), 'w') as cfg,
               open(os.path.join(root_dir, config['labels']), 'wb') as lf
               ):
            json.dump(config, cfg)
            np.save(lf, self.labels)

    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> 'LLMLabels':
        with open(os.path.join(root_dir, DATA_DIR, filename), 'r') as cfg:
            config = json.load(cfg)
        return LLMLabels(llm=LLMType.from_str(config['llm']), labels=np.load(os.path.join(root_dir, config['labels'])) )
    
    def __call__(self, root_dir: os.PathLike, filename: str) -> 'LLMLabels':
        self.__filename = filename
        self.__root_dir = root_dir
        return self 
    
    def __enter__(self) -> 'LLMLabels':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__root_dir is None or self.__filename is None:
            print("Warning: cannot dump LLMLabels on exit")
        else:
            self.dump(self.__root_dir, self.__filename)

class Dataset(Dumpable, JSONifiable):
    def __init__(self, id: DatasetID, text_field: str, label_field: str):
        self.id = id
        self.text_field = text_field
        self.label_field = label_field
        self.__dataset = None
        self.__annotations = None
    
    @property
    def dataset(self) -> 'DatasetView': 
        return DatasetView(self)
    
    def to_small_text(self) -> datasets.DatasetDict:
        if self.__dataset is not None:
            return self.__dataset
        if self.id.subset is None:
            self.__dataset = datasets.load_dataset(self.id.path)
        else:
            self.__dataset = datasets.load_dataset(self.id.path, self.id.subset)
        self.__dataset = Dataset.__standardize_dataset(self.__dataset, self.text_field, self.label_field)
        return self.__dataset 
    
    @property
    def annotations(self) ->  dict[LLMType, LLMLabels]:
        if self.__annotations is not None:
            return self.__annotations
        self.__annotations = {}
        dataset_file = os.path.join(self.root_dir, DATA_DIR, self.get_config_filename())
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r') as df:
                config = json.load(df)
            self.__load_annotations(config)
        return self.__annotations
    
    def __load_annotations(self, config: dict):
        self.__annotations = {
            LLMType.from_str(llm_str): LLMLabels.load(self.root_dir, label_file)
            for llm_str, label_file in config['annotations'].items()
        }

    def get_config_filename(self, external_id: str | None = None) -> str:
        if external_id is not None:
            raise ValueError("external_id can not be provided for Dataset")
        return f"{str(self.id)}.json"
    
    def get_config(self, external_id: str | None = None) -> dict:
        if external_id is not None:
            raise ValueError("external_id can not be provided for Dataset")
        config = {
            'id': str(self.id),
            'annotations': {
                str(llm): os.path.join(DATA_DIR, labels.get_config_filename(str(self.id))) 
                for llm, labels in self.annotations.items()
            } if self.__annotations is not None else {},
            'text_field': self.text_field,
            'label_field': self.label_field,
        }       
        return config
    
    def dump(self, root_dir: os.PathLike, filename: str | None = None):
        config = self.get_config()
        with open(os.path.join(
                        root_dir, 
                        DATA_DIR, 
                        filename if filename is not None else self.get_config_filename()
                    ), 'w') as df:
            json.dump(config, df)
        for labels in self.annotations.values():  
            labels.dump(root_dir, str(self.id))

    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> 'Dataset':
        with open(os.path.join(root_dir, DATA_DIR, filename), 'r') as df:
            config = json.load(df)
        dataset = Dataset(
            id=DatasetID.from_str(config['id']),
            text_field=config['text_field'],
            label_field=config['label_field'],
        )
        dataset.__load_annotations(config)
        return dataset
    
    def __call__(self, root_dir: os.PathLike, filename: str | None = None) -> 'Dataset':
        self.__root_dir = root_dir
        self.__filename = filename
        return self
    
    def __enter__(self) -> 'Dataset':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__root_dir is None:
            print("Warning: cannot dump Dataset on exit")
        else:
            self.dump(self.__root_dir, self.__filename)

    @staticmethod
    def __standardize_dataset(dataset: datasets.DatasetDict, text_field: str, label_field: str) -> datasets.DatasetDict:
        return dataset.map(
            lambda ex: {
                "text": ex[text_field],
                "label": ex[label_field],
            },
            remove_columns=dataset["train"].column_names,
        )

    def to_json(self) -> dict:
        return {
            'id': str(self.id),
            'text_field': self.text_field,
            'label_field': self.label_field,
        }
    
    @staticmethod
    def from_json(data: dict) -> 'Dataset':
        return Dataset(
            id=DatasetID.from_str(data['id']),
            text_field=data['text_field'],
            label_field=data['label_field'],
        )
    
class Pool(Dumpable):
    def __init__(self, indices: Indices, dataset: Dataset | None):
        self.indices = indices
        self.__dataset = dataset
        self.__pool = None
        self.__annotations = None
    
    @property
    def base_dataset(self) -> Dataset | None:
        return self.__dataset
    
    def to_small_text(self) -> datasets.DatasetDict:
        if self.__pool is None:
            self.__pool = self.__dataset.dataset.select(self.indices.indices.tolist())
        return self.__pool

    @property
    def pool(self) -> 'DatasetView':
        return DatasetView(self)
    
    @property
    def size(self) -> int:
        return len(self.indices)
    
    @property
    def annotations(self) ->  dict[LLMType, LLMLabels]:
        if self.__annotations is not None:
            return self.__annotations
        if self.__dataset is not None:
            all_annotations = self.__dataset.annotations
            self.__annotations = {}
            for llm, labels in all_annotations.items():
                pool_labels = labels.labels[self.indices.indices]
                self.__annotations[llm] = LLMLabels(llm=llm, labels=pool_labels)
            return self.__annotations
        raise ValueError("Cannot get annotations without base dataset")

    def get_config(self, external_id: str) -> dict:
        return {
            'indices_file': os.path.join(DATA_DIR, self.indices.get_config_filename(external_id)),
            'pool_files': {     
                key: os.path.join(DATA_DIR, f"{external_id}_pool_{key}.dat") for key in self.pool.dataset
                },
            'dataset_id': str(self.__dataset.get_id()) if self.__dataset is not None else None,
            'annotations': {
                str(llm): os.path.join(DATA_DIR, labels.get_config_filename(external_id)) 
                for llm, labels in self.annotations.items()
            } if self.__annotations is not None else {},
        }
    
    def get_config_filename(self, external_id : str) -> str:
        return f"{external_id}_pool.json"
    
    def __str__(self) -> str:
        return self.indices.repr

    def dump(self, root_dir: os.PathLike, external_id: str, filename: str | None = None):
        config = self.get_config(external_id)
        with open(os.path.join(
                root_dir, 
                DATA_DIR, 
                filename if filename is not None else self.get_config_filename(external_id)
                ), 'w') as cfg:
            json.dump(config, cfg)
        self.indices.dump(root_dir, external_id)
        for key, file in config['pool_files'].items():
            self.pool.dataset[key].save_to_disk(os.path.join(root_dir, file))
        for llm, file in config['annotations'].items():
            self.annotations[LLMType.from_str(llm)].dump(root_dir, external_id)

    @staticmethod
    def load(root_dir: os.PathLike, filename: str, dataset: Dataset | None = None) -> 'Pool':
        config_path = filename if os.path.isabs(filename) else os.path.join(root_dir, filename)
        with open(config_path, 'r') as cfg:
            config = json.load(cfg)
        indices_file = config['indices_file']
        if indices_file.startswith(f"{DATA_DIR}{os.sep}"):
            indices_file = indices_file[len(DATA_DIR) + 1:]
        indices = Indices.load(root_dir, indices_file)
        pool_data = datasets.DatasetDict()
        for key, file in config['pool_files'].items():
            pool_data[key] = datasets.Dataset.load_from_disk(os.path.join(root_dir, file))
        pool = Pool(indices=indices, dataset=dataset)
        pool.__pool = pool_data
        pool.__annotations = {
            LLMType.from_str(llm_str): LLMLabels.load(
                root_dir,
                label_file[len(DATA_DIR) + 1:] if label_file.startswith(f"{DATA_DIR}{os.sep}") else label_file
            )
            for llm_str, label_file in config['annotations'].items()
        }
        return pool
    
    def __call__(self, root_dir: os.PathLike, external_id: str, filename: str | None = None) -> 'Pool':
        self.__root_dir = root_dir
        self.__external_id = external_id
        self.__filename = filename
        return self
    
    def __enter__(self) -> 'Pool':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__root_dir is None or self.__external_id is None:
            print("Warning: cannot dump Pool on exit")
        else:
            self.dump(self.__root_dir, self.__external_id, self.__filename)


class DatasetView:
    def __init__(self, base: Dataset | Pool):
        self.base = base

    @property
    def dataset(self) -> datasets.DatasetDict:
        return self.base.to_small_text()
    
    @property
    def annotations(self) ->  dict[LLMType, LLMLabels]:
        return self.base.annotations
    
    @property
    def train(self) -> datasets.Dataset:
        return self.dataset["train"]
    
    @property
    def validation(self) -> datasets.Dataset:
        return self.dataset.get("validation", self.dataset["test"])
    
    @staticmethod
    def to_transformers_dataset(subset: datasets.Dataset, tokenizer: BertTokenizerFast, num_classes: int, max_length: int = 128) -> TransformersDataset:
        texts = subset["text"]
        labels = np.array(subset["label"], dtype=np.int64)
        target_labels = np.arange(num_classes, dtype=np.int64)

        return TransformersDataset.from_arrays(
            texts,
            labels,
            tokenizer,
            target_labels=target_labels,
            max_length=max_length,
        )


class Datasets:
    def __init__(self, *datasets: Dataset):
        self.datasets = datasets
        
class DataDatabase:
    def __init__(self):
        self.datasets: Datasets 
        self.experiments: 'Experiments'

        ...
    
    # def loadExperiment()
