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

class Indices:
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

    def dump(self, root_dir: os.PathLike, file_name: str):
        config = {
            'hardcoded': self.repr is None, 
        }
        if self.repr is not None:
            config['repr'] = self.repr
        else:
            indices_file =  os.path.join(root_dir, DATA_DIR, f"{file_name}_indices.dat") 
            config['indices_file'] = indices_file
            np.save(indices_file, self.indices)

        with open(os.path.join(root_dir, DATA_DIR, f"{file_name}_indices.json"), 'w') as cfg:
            json.dump(config, cfg)

    @staticmethod
    def load(root_dir: os.PathLike, file_name: str) -> 'Indices':
        with open(os.path.join(root_dir, DATA_DIR, f"{file_name}_indices.json"), 'r') as cfg:
            config = json.load(cfg)
        if config['hardcoded']:
            indices = np.load(config['indices_file'])
            return Indices(indices=indices, repr=None)
        else:
            return Indices.from_seed(**config['repr'])
@dataclasses.dataclass
class DatasetID:
    path: str
    subset: str | None = None
    base: 'DatasetID' | None = None
    indices: Indices | None = None

    def __str__(self):
        if self.base is None:
            if self.subset is None:
                return self.path
            return f"{self.path.replace('/', '_')} {self.subset}"
        else:
            return f"Pool[{str(self.base)}#{self.indices.repr}]"
    
    @staticmethod
    def from_str(id_str: str) -> 'DatasetID':
        if id_str.startswith('Pool['):
            inner = id_str[len('Pool['):-1]
            base_str, indices_str = inner.split('#')
            base = DatasetID.from_str(base_str)
            indices_repr = json.loads(indices_str.replace("'", '"'))
            indices = Indices.from_seed(**indices_repr)
            return DatasetID(path='', subset=None, base=base, indices=indices)
        if not id_str.endswith('None'):
            parts = id_str.split(' ')
            path = parts[0].replace('_', '/')
            subset = parts[-1]
            return DatasetID(path=path, subset=subset)
        else:
            return DatasetID(path=id_str, subset=None)
    
class LLMType(enum.Enum):
    GIGACHAT = enum.auto()

    STR_MAPPINGS = {
        GIGACHAT: "gigachat",
    }

    REVERSE_STR_MAPPINGS = {v: k for k, v in STR_MAPPINGS.items()}

    def to_str(self) -> str:
        return self.STR_MAPPINGS[self]
    
    @classmethod
    def from_str(cls: 'LLMType', strategy_str: str) -> 'LLMType':
        return cls.REVERSE_STR_MAPPINGS[strategy_str]


@dataclasses.dataclass
class LLMLabels:
    llm: LLMType
    labels: npt.NDArray[np.int64]

    def dump(self, root_dir: os.PathLike, ID: str):
        label_file =  os.path.join(root_dir, DATA_DIR, f"{LLMLabels.id_to_file_name(ID, self.llm)}.dat") 
        config = {
            'llm': self.llm.to_str(), 
            'labels': label_file, 
            'ID': ID, 
        }
        with open(os.path.join(root_dir, DATA_DIR, f"{LLMLabels.id_to_file_name(ID, self.llm)}.json"), 'w') as cfg, open(label_file, 'wb') as lf:
            json.dump(config, cfg)
            np.save(lf, self.labels)

    @staticmethod
    def load(root_dir: os.PathLike, path: str) -> 'LLMLabels':
        with open(os.path.join(root_dir, DATA_DIR, f"{path}.json"), 'r') as cfg:
            config = json.load(cfg)
        return LLMLabels(llm=LLMType.from_str(config['llm']), labels=np.load(config['labels']) )
    
    @staticmethod
    def id_to_file_name(ID: str, llm: LLMType) -> str:
        return f"{ID}_{llm.to_str()}"

class Dataset:
    def __init__(self, id: DatasetID, text_field: str, label_field: str, root_dir: str | None = None):
        self.id = id
        self.text_field = text_field
        self.label_field = label_field
        self.__dataset = None
        self.__annotations = None
        self.root_dir = root_dir if root_dir is not None else os.getcwd()

    def get_id(self) -> str:
        return str(self.id)
    
    def dump(self, root_dir: os.PathLike):
        str_id = self.get_id()
        dataset_file = os.path.join(root_dir, DATA_DIR, f"{str_id}.json")
        config = {
            'id': str_id,
            'annotations': {llm.to_str(): LLMLabels.id_to_file_name(str_id, llm) for llm in self.annotations.keys()} if self.__annotations is not None else {},
        }    

        if self.id.base is not None and self.id.indices is not None:
            config['base'] = str(self.id.base)
            config['base_file'] = os.path.join(DATA_DIR, f"{str(self.id.base)}.json")
            config['indices_file'] = os.path.join(DATA_DIR, f"{str_id}_indices.json")
            Dataset(id=self.id.base, text_field=self.text_field, label_field=self.label_field, root_dir=self.root_dir).dump(root_dir)
            self.id.indices.dump(root_dir, str_id)
    
        with open(dataset_file, 'w') as df:
            json.dump(config, df)
        for _, labels in self.annotations.items():  
            labels.dump(root_dir, str_id)
        
    def get_filename(self) -> str:
        return os.path.join(DATA_DIR, f"{self.get_id()}.json")

    @property
    def annotations(self) ->  dict[LLMType, LLMLabels]:
        if self.__annotations is None:
            self.__annotations = {}
            dataset_file = os.path.join(self.root_dir, DATA_DIR, f"{str(self.id)}.json")
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r') as df:
                    config = json.load(df)
                for llm_str, label_file in config['annotations'].items():
                    llm = LLMType.from_str(llm_str)
                    self.__annotations[llm] = LLMLabels.load(self.root_dir, label_file)
        return self.__annotations

    @property
    def dataset(self) -> datasets.Dataset:
        if self.__dataset is None:
            if self.id.base is not None and self.id.indices is not None:
                base_dataset = Dataset(self.id.base, self.text_field, self.label_field, self.root_dir)
                self.__dataset = base_dataset.dataset.select(self.id.indices.indices.tolist())
                return self.__dataset
            if self.id.subset is None:
                self.__dataset = datasets.load_dataset(self.id.path, trust_remote_code=True)
            else:
                self.__dataset = datasets.load_dataset(self.id.path, self.id.subset, trust_remote_code=True)
            self.__dataset = Dataset.__standardize_dataset(self.__dataset, self.text_field, self.label_field)
        return self.__dataset
    
    def __repr__(self):
        return f"Dataset(path={self.id.path}, subset={self.id.subset})"

    @staticmethod
    def __standardize_dataset(dataset: datasets.Dataset, text_field: str, label_field: str):
        return dataset.map(
            lambda ex: {
                "text": ex[text_field],
                "label": ex[label_field],
            },
            remove_columns=dataset["train"].column_names,
        )
    
    @property
    def train(self) -> datasets.Dataset:
        return self.dataset["train"]
    
    @property
    def validation(self) -> datasets.Dataset:
        return self.dataset.get("validation", self.dataset["test"])

class Pool:
    def __init__(self, indices: Indices, dataset: Dataset | None):
        self.indices = indices
        self.__dataset = dataset
        self.__pool = None
    
    @property
    def pool(self) -> Dataset:
        if self.__pool is None:
            self.__pool = Dataset(
                id=DatasetID(
                    path='',
                    subset=None,
                    base=self.__dataset.id,
                    indices=self.indices,
                ),
                text_field=self.__dataset.text_field,
                label_field=self.__dataset.label_field,
                root_dir=self.__dataset.root_dir,
            )            
        return self.__pool
    
    @property
    def size(self) -> int:
        return len(self.indices)
    
    def __str__(self):
        return self.indices.repr

    def dump(self, root_dir: os.PathLike, file_name: str):
        config = {
            'indices_file': f"{file_name}_indices.json",
            'pool_files': {key: f"{file_name}_pool_{key}.dat" for key in self.pool.dataset},
            'dataset_id': str(self.__dataset.get_id()) if self.__dataset is not None else None,
        }
        with open(os.path.join(root_dir, DATA_DIR, f"{file_name}_pool.json"), 'w') as cfg:
            json.dump(config, cfg)
        self.indices.dump(root_dir, file_name)
        for key, file in config['pool_files'].items():
            np.save(os.path.join(root_dir, DATA_DIR, file), self.pool.dataset[key])

    @staticmethod
    def load(root_dir: os.PathLike, file_name: str, dataset: Dataset | None = None) -> 'Pool':
        with open(os.path.join(root_dir, DATA_DIR, f"{file_name}_pool.json"), 'r') as cfg:
            config = json.load(cfg)
        indices = Indices.load(root_dir, file_name)
        pool_data = datasets.DatasetDict()
        for key, file in config['pool_files'].items():
            data_array = np.load(os.path.join(root_dir, DATA_DIR, file))
            pool_data[key] = datasets.Dataset.from_dict({key: data_array})
        pool = Pool(indices=indices, dataset=dataset)
        pool.__pool = Dataset(
            id=DatasetID(
                path='',
                subset=None,
                base=dataset.id if dataset is not None else DatasetID.from_str(config['dataset_id']),
                indices=indices,
            ),
            text_field=dataset.text_field if dataset is not None else 'text',
            label_field=dataset.label_field if dataset is not None else 'label',
            root_dir=root_dir,
        )
        pool.__pool.__dataset = pool_data
        return pool

class Datasets:
    def __init__(self, *datasets: Dataset):
        self.datasets = datasets
        
class DataDatabase:
    def __init__(self):
        self.datasets: Datasets 
        self.experiments: 'Experiments'

        ...
    
    # def loadExperiment()

def to_transformers_dataset(tokenizer: BertTokenizerFast, subset: datasets.Dataset, num_classes: int, max_length: int = 128) -> TransformersDataset:
    texts = subset["text"]
    labels = np.array(subset["label"], dtype=np.int64)
    target_labels = np.arange(num_classes, dtype=np.int64)

    ds = TransformersDataset.from_arrays(
        texts,
        labels,
        tokenizer,
        target_labels=target_labels,
        max_length=max_length,
    )
    return ds

