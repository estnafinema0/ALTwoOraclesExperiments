import dataclasses
from strategies import ActiveLearningStrategy, ColdStartStrategy
import datasets
import enum
import numpy as np
import numpy.typing as npt
import os
import json

DATA_DIR = 'data'

@dataclasses.dataclass
class DatasetID:
    path: str
    subset: str | None = None

    def __str__(self):
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
    def __init__(self, id: DatasetID, root_dir: str | None = None):
        self.id = id
        self.__dataset = None
        self.__annotations = None
        self.root_dir = root_dir if root_dir is not None else os.getcwd()

    def dump(self, root_dir: os.PathLike):
        str_id = str(self.id)
        dataset_file = os.path.join(root_dir, DATA_DIR, f"{str_id}.json")
        config = {
            'id': str_id,
            'annotations': {llm.to_str(): LLMLabels.id_to_file_name(str_id, llm) for llm in self.annotations.keys()} if self.__annotations is not None else {},
        }    
        with open(dataset_file, 'w') as df:
            json.dump(config, df)
        for _, labels in self.annotations.items():  
            labels.dump(root_dir, str_id)

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
            if self.id.subset is None:
                self.__dataset = datasets.load_dataset(self.id.path, trust_remote_code=True)
            else:
                self.__dataset = datasets.load_dataset(self.id.path, self.id.subset, trust_remote_code=True)
        return self.__dataset
    
    def __repr__(self):
        return f"Dataset(path={self.id.path}, subset={self.id.subset})"

 

def subsample_pool(hf_train_split: Dataset, pool_size: int, seed: int):
    """Сэмплируем pool_size из train."""
    n = len(hf_train_split)
    pool_size = min(pool_size, n)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    selected = indices[:pool_size]
    return hf_train_split.select(selected.tolist()), selected

class Indices:
    ...

class Pool:
    def __init__(self, root_dir: os.PathLike, dataset_id: DatasetID, indices: npt.NDArray[np.int64]):
        self.root_dir = root_dir
        self.dataset_id = dataset_id
        self.indices = indices
        self.__dataset = None
        self.__pool = None
    
    @property
    def pool(self) -> datasets.Dataset:
        if self.__pool is None:
            if self.__dataset is None:
                dataset = Dataset(self.dataset_id, root_dir=self.root_dir)
                self.__dataset = dataset.dataset['train']
            self.__pool = self.__dataset.select(self.indices.tolist())
        return self.__pool
    
    @property
    def size(self) -> int:
        return len(self.indices)
    


class Datasets:
    def __init__(self, *datasets: Dataset):
        self.datasets = datasets
        
class DataDatabase:
    def __init__(self):
        self.datasets: Datasets 
        self.experiments: 'Experiments'

        ...
    
    # def loadExperiment()
        
@dataclasses.dataclass
class Experiment:
    seed: int
    dataset: Dataset
    split: int 
    cold_start_strategy: ColdStartStrategy
    active_learning_strategy: ActiveLearningStrategy
    macro_f1: float
    accuracy: float

    @staticmethod
    def load_experiment(format: 'ExperimentFormat', experiment_config: 'ExperimentConfig') -> 'Experiment':
        ...

    @property
    def budget(self) -> int:
        return self.cold_start_strategy.budget + self.active_learning_strategy.budget
