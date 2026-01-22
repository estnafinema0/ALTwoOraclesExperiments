from utils import Stringifiable
import database

import small_text
from small_text.query_strategies import QueryStrategy
from small_text.integrations.pytorch.query_strategies import BADGE
from small_text.query_strategies.bayesian import BALD
from small_text.query_strategies.strategies import RandomSampling, LeastConfidence
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text import PoolBasedActiveLearner
from small_text.integrations.transformers.datasets import TransformersDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import numpy.typing as npt

import enum
import datasets
import dataclasses
import datetime
from abc import ABC, abstractmethod
import time

def evaluate_on_test(active_learner: PoolBasedActiveLearner, test_dataset: TransformersDataset) -> tuple[float, float]:
    y_pred = active_learner.classifier.predict(test_dataset)
    y_true = test_dataset.y

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1

class QueryStrategyType(ABC, Stringifiable):
    @abstractmethod
    def get_parameters(self) -> dict[str, any]:
        pass
    
    @abstractmethod
    def query_strategy_name(self) -> str:
        pass
    
    @abstractmethod
    def query_strategy_class(self) -> QueryStrategy:
        pass

    @abstractmethod
    def run_loop(self, pool: database.DatasetView, active_learner: PoolBasedActiveLearner, indices_labeled: npt.NDArray[np.int64], test_dataset: TransformersDataset, n_iterations: int, batch_size: int) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        pass
    
    def __str__(self) -> str:
        return self.query_strategy_name()

    @abstractmethod
    @staticmethod
    def from_str(s: str) -> 'QueryStrategyType':
        pass

class SimpleQueryStrategyType(enum.Enum, Stringifiable):
    RANDOM = enum.auto()
    LEAST_CONFIDENCE = enum.auto()
    BALD = enum.auto()
    BADGE = enum.auto()

    STR_MAPPINGS = {
        RANDOM: "random",
        LEAST_CONFIDENCE: "least_confidence",
        BALD: "bald",
        BADGE: "badge",
    }
    REVERSE_STR_MAPPINGS = {v: k for k, v in STR_MAPPINGS.items()}
    QUERY_STRATEGY_CLASS_MAPPINGS = {
        RANDOM: RandomSampling,
        LEAST_CONFIDENCE: LeastConfidence,
        BALD: small_text.integrations.pytorch.query_strategies.BALD,
        BADGE: small_text.query_strategies.bayesian.BADGE,
    }
    REVERSE_QUERY_STRATEGY_CLASS_MAPPINGS = {v: k for k, v in QUERY_STRATEGY_CLASS_MAPPINGS.items()}

    def __str__(self) -> str:
        return self.STR_MAPPINGS[self]
    
    @classmethod
    def from_str(cls: 'SimpleQueryStrategyType', strategy_str: str) -> 'SimpleQueryStrategyType':
        return cls.REVERSE_STR_MAPPINGS[strategy_str]
    
    def to_query_strategy(self, num_classes: int) -> QueryStrategy:
        return SimpleQueryStrategyType.__make_query_strategy_instance(self, num_classes)

    @classmethod
    def from_query_strategy(cls: 'SimpleQueryStrategyType', strategy: QueryStrategy) -> 'SimpleQueryStrategyType':
        for key, val in cls.REVERSE_QUERY_STRATEGY_CLASS_MAPPINGS.items():
            if isinstance(strategy, key):
                return val
        raise ValueError("Unknown QueryStrategy class")

    @staticmethod
    def __make_query_strategy_instance(strategy: 'SimpleQueryStrategyType', num_classes: int) -> QueryStrategy:
        if strategy == BALD:
            return BALD(dropout_samples=10)
        elif strategy == BADGE:
            return BADGE(num_classes=num_classes)
        else:
            return SimpleQueryStrategyType.QUERY_STRATEGY_CLASS_MAPPINGS[strategy]()

class QueryStrategySimple(QueryStrategyType):
    def __init__(self, simple_strategy: SimpleQueryStrategyType):
        super().__init__()
        self.__strategy = simple_strategy

    def get_parameters(self) -> dict[str, any]:
        return {}
    
    def query_strategy_name(self) -> str:
        return self.__strategy.to_str()
    
    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
        return self.__strategy.to_query_strategy(num_classes=num_classes)
    
    def run_loop(self, pool: database.DatasetView, active_learner: PoolBasedActiveLearner, indices_labeled: npt.NDArray[np.int64], test_dataset: TransformersDataset, n_iterations: int, batch_size: int) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        start = time.perf_counter()
        for _ in range(n_iterations):
            queried_indices = active_learner.query(num_samples=batch_size)
            y_queried = pool.train.y[queried_indices] # TODO: implement querying logic from different oracles
            active_learner.update(y_queried)
            indices_labeled = np.concatenate([indices_labeled, queried_indices])

        end = time.perf_counter()

        acc, macro_f1 = evaluate_on_test(active_learner, test_dataset)
        duration = datetime.timedelta(seconds=end - start)
        return acc, macro_f1, duration, indices_labeled
    
    @staticmethod
    def from_str(s: str) -> 'QueryStrategySimple':
        return QueryStrategySimple(SimpleQueryStrategyType.from_str(s))

class MockQueryStrategyType(QueryStrategyType):
    def get_parameters(self) -> dict[str, any]:
        return {}
    
    def query_strategy_name(self) -> str:
        return "mock"
    
    def query_strategy_class(self) -> QueryStrategy:
        raise NotImplementedError("MockQueryStrategyType does not implement query_strategy_class")
    
    def run_loop(self, pool: database.DatasetView, active_learner: PoolBasedActiveLearner, indices_labeled: npt.NDArray[np.int64], test_dataset: TransformersDataset, n_iterations: int, batch_size: int) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        start = time.perf_counter()
        end = time.perf_counter()
        acc, macro_f1 = evaluate_on_test(active_learner, test_dataset)
        duration = datetime.timedelta(seconds=end - start)
        return acc, macro_f1, duration, indices_labeled
    
    @staticmethod
    def from_str(s: str) -> 'MockQueryStrategyType':
        if s != "mock":
            raise ValueError("Invalid string for MockQueryStrategyType")
        return MockQueryStrategyType()
    
    
@dataclasses.dataclass
class ColdStartStrategy(Stringifiable):
    query_strategy: QueryStrategyType
    batch_size: int # Размер выборки для обучения на одной итерации
    
    @property # TODO: make multiple iterations possible
    def budget(self) -> int:
        return self.batch_size
    
    @property
    def n_iterations(self) -> int:
        return 1
    
    def __str__(self) -> str:
        return f"{self.query_strategy.query_strategy_name()}_{self.budget}_{self.batch_size}"
    
    @staticmethod
    def from_str(strategy_str: str) -> 'ColdStartStrategy':
        parts = strategy_str.split('_')
        if len(parts) != 3:
            raise ValueError("Invalid ColdStartStrategy string representation")
        query_strategy = SimpleQueryStrategyType.from_str(parts[0])
        budget = int(parts[1])
        batch_size = int(parts[2])
        if budget != batch_size:
            raise ValueError("Inconsistent budget and batch_size for ColdStartStrategy")
        return ColdStartStrategy(
            query_strategy=QueryStrategySimple(query_strategy),
            batch_size=batch_size
        )
    
    @staticmethod
    def from_budget(query_strategy: QueryStrategyType, budget: int) -> 'ColdStartStrategy':
        return ColdStartStrategy(
            query_strategy=query_strategy,
            batch_size=budget
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColdStartStrategy):
            return False
        return (self.query_strategy.query_strategy_name() == other.query_strategy.query_strategy_name() and
                self.batch_size == other.batch_size and self.budget == other.budget
                )
    
    def __hash__(self):
        return hash((self.query_strategy.query_strategy_name(), self.batch_size, self.budget))

class ActiveLearningStrategy(Stringifiable):
    def __init__(
        self,
        query_strategy: QueryStrategyType,
        batch_size: int, # Размер выборки для обучения на одной итерации
        budget: int, # Общий бюджет на активное обучение
    ):
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.budget = budget

    @property
    def n_iterations(self) -> int:
        return self.budget // self.batch_size + (1 if self.budget % self.batch_size > 0 else 0)

    def __str__(self) -> str:
        return f"{self.query_strategy.query_strategy_name()}_{self.budget}_{self.n_iterations}_{self.batch_size}"

    @staticmethod
    def from_str(strategy_str: str) -> 'ActiveLearningStrategy':
        parts = strategy_str.split('_')
        if len(parts) != 4:
            raise ValueError("Invalid ActiveLearningStrategy string representation")
        query_strategy = SimpleQueryStrategyType.from_str(parts[0])
        budget = int(parts[1])
        n_iterations = int(parts[2])
        step_size = int(parts[3])
        if budget != step_size * (n_iterations - 1) + (budget - step_size * (n_iterations - 1)):
            raise ValueError("Inconsistent budget, step_size and n_iterations")
        return ActiveLearningStrategy(
            query_strategy=QueryStrategySimple(query_strategy),
            batch_size=step_size,
            budget=budget
        )
    
    def batch_size_at(self, iteration: int) -> int:
        if iteration < 0 or iteration >= self.n_iterations:
            raise ValueError("Invalid iteration number")
        if iteration < self.n_iterations - 1:
            return self.batch_size
        else:
            return self.budget - self.batch_size * (self.n_iterations - 1)
        

class ComposeStrategyWrapper(QueryStrategy):
    def __init__(self, al_strategy: ActiveLearningStrategy, cs_strategy: ColdStartStrategy):
        self.al_strategy = al_strategy
        self.cs_strategy = cs_strategy
        self.__budget_used = 0
        self.__al_strategy = None
        self.__cs_strategy = None

    @property
    def cold_start_strategy(self)-> QueryStrategy:
        if self.__cs_strategy is None:
            self.__cs_strategy = self.cs_strategy.query_strategy.query_strategy_class()
        return self.__cs_strategy
    
    @property
    def active_learning_strategy(self) -> QueryStrategy:
        if self.__al_strategy is None:
            self.__al_strategy = self.al_strategy.query_strategy.query_strategy_class()
        return self.__al_strategy

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        if self.__budget_used + n < self.cs_strategy.budget:
            strategy = self.cold_start_strategy
        elif self.__budget_used + n > self.cs_strategy.budget:
            raise ValueError("Requested more samples than available in cold start pool")
        elif self.__budget_used + n == self.cs_strategy.budget:
            strategy = self.cold_start_strategy  
        else:
            strategy = self.active_learning_strategy    
        self.__budget_used += n
        return strategy.query(
                clf,
                dataset,
                indices_unlabeled,
                indices_labeled,
                y,
                n
            )   
