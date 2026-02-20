from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from small_text import PoolBasedActiveLearner
    from small_text.integrations.transformers.datasets import TransformersDataset

from utils import EnumABCMeta
from storage import Stringifiable
import database

from small_text.query_strategies import QueryStrategy
from small_text.integrations.pytorch.query_strategies import BADGE as st_BADGE
from small_text.query_strategies.bayesian import BALD as st_BALD
import numpy as np
import numpy.typing as npt

import enum
from typing import Self
import datasets
import dataclasses
import datetime
from abc import ABC, abstractmethod
import time
import itertools
import re


def evaluate_on_test(
    active_learner: 'PoolBasedActiveLearner', test_dataset: 'TransformersDataset'
) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = active_learner.classifier.predict(test_dataset)
    y_true = test_dataset.y

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return acc, macro_f1


class QueryStrategyType(Stringifiable):
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
    def run_loop(
        self,
        pool: 'database.Pool',
        active_learner: 'PoolBasedActiveLearner',
        indices_labeled: npt.NDArray[np.int64],
        test_dataset: 'TransformersDataset',
        n_iterations: int,
        batch_size: int,
    ) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        pass

    @classmethod
    def __get_all_subclasses(cls) -> list:
        return cls.__subclasses__() + list(
            itertools.chain.from_iterable(subcls.__get_all_subclasses() for subcls in cls.__subclasses__())
        )

    def __str__(self) -> str:
        return self.query_strategy_name()

    @staticmethod
    @abstractmethod
    def from_str(s: str) -> 'QueryStrategyType':
        pass

    @classmethod
    def factory_from_str(cls, s: str) -> 'QueryStrategyType':
        for subclass in cls.__get_all_subclasses():
            try:
                return subclass.from_str(s)
            except ValueError:
                continue
        raise ValueError(f'Unknown QueryStrategyType string: {s}')

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


class SimpleQueryStrategyType(Stringifiable, enum.StrEnum, metaclass=EnumABCMeta):
    RANDOM = enum.auto()
    LEAST_CONFIDENCE = enum.auto()
    BALD = enum.auto()
    BADGE = enum.auto()

    @classmethod
    def query_strategy_class_mappings(cls) -> dict[Self, type[QueryStrategy]]:
        from small_text.query_strategies.strategies import RandomSampling, LeastConfidence

        return {
            cls.RANDOM: RandomSampling,
            cls.LEAST_CONFIDENCE: LeastConfidence,
            cls.BALD: st_BALD,
            cls.BADGE: st_BADGE,
        }

    @classmethod
    def reverse_query_strategy_class_mappings(cls) -> dict[type[QueryStrategy], Self]:
        return {v: k for k, v in cls.query_strategy_class_mappings().items()}

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls: 'SimpleQueryStrategyType', strategy_str: str) -> 'SimpleQueryStrategyType':
        if strategy_str not in SimpleQueryStrategyType:
            raise ValueError(f'Unknown SimpleQueryStrategyType string: {strategy_str}')
        return SimpleQueryStrategyType(strategy_str)

    def to_query_strategy(self, num_classes: int) -> QueryStrategy:
        return SimpleQueryStrategyType.__make_query_strategy_instance(self, num_classes)

    @classmethod
    def from_query_strategy(cls: 'SimpleQueryStrategyType', strategy: QueryStrategy) -> 'SimpleQueryStrategyType':
        for key, val in cls.reverse_query_strategy_class_mappings().items():
            if isinstance(strategy, key):
                return val
        raise ValueError('Unknown QueryStrategy class')

    @staticmethod
    def __make_query_strategy_instance(strategy: 'SimpleQueryStrategyType', num_classes: int) -> QueryStrategy:
        if strategy == SimpleQueryStrategyType.BALD:
            return st_BALD(dropout_samples=10)
        elif strategy == SimpleQueryStrategyType.BADGE:
            return st_BADGE(num_classes=num_classes)
        else:
            return SimpleQueryStrategyType.query_strategy_class_mappings()[strategy]()


@Stringifiable.make_stringifiable()
class QueryStrategySimple(QueryStrategyType, Stringifiable):
    def __init__(self, simple_strategy: SimpleQueryStrategyType):
        super().__init__()
        self.__strategy = simple_strategy

    def get_parameters(self) -> dict[str, any]:
        return {}

    def query_strategy_name(self) -> str:
        return str(self.__strategy)

    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
        return self.__strategy.to_query_strategy(num_classes=num_classes)

    def run_loop(
        self,
        pool: 'database.Pool',
        active_learner: 'PoolBasedActiveLearner',
        indices_labeled: npt.NDArray[np.int64],
        test_dataset: 'TransformersDataset',
        n_iterations: int,
        batch_size: int,
    ) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        start = time.perf_counter()
        for _ in range(n_iterations):
            queried_indices = active_learner.query(num_samples=batch_size)
            y_queried = pool.y[queried_indices]  # TODO: implement querying logic from different oracles
            active_learner.update(y_queried)
            indices_labeled = np.concatenate([indices_labeled, queried_indices])

        end = time.perf_counter()

        acc, macro_f1 = evaluate_on_test(active_learner, test_dataset)
        duration = datetime.timedelta(seconds=end - start)
        return acc, macro_f1, duration, indices_labeled

    def __str__(self) -> str:
        return str(self.__strategy)

    @staticmethod
    def from_str(s: str) -> 'QueryStrategySimple':
        return QueryStrategySimple(SimpleQueryStrategyType.from_str(s))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QueryStrategySimple) and self.__strategy == other.__strategy

    def __hash__(self) -> int:
        return hash(self.__strategy)


@Stringifiable.make_stringifiable()
class MockQueryStrategyType(QueryStrategyType, Stringifiable):
    def get_parameters(self) -> dict[str, any]:
        return {}

    def query_strategy_name(self) -> str:
        return 'mock'

    def query_strategy_class(self) -> QueryStrategy:
        raise NotImplementedError('MockQueryStrategyType does not implement query_strategy_class')

    def run_loop(
        self,
        pool: 'database.Pool',
        active_learner: 'PoolBasedActiveLearner',
        indices_labeled: npt.NDArray[np.int64],
        test_dataset: 'TransformersDataset',
        n_iterations: int,
        batch_size: int,
    ) -> tuple[float, float, datetime.timedelta, npt.NDArray[np.int64]]:
        start = time.perf_counter()
        end = time.perf_counter()
        acc, macro_f1 = evaluate_on_test(active_learner, test_dataset)
        duration = datetime.timedelta(seconds=end - start)
        return acc, macro_f1, duration, indices_labeled

    def __str__(self) -> str:
        return 'mock'

    @staticmethod
    def from_str(s: str) -> 'MockQueryStrategyType':
        if s != 'mock':
            raise ValueError('Invalid string for MockQueryStrategyType')
        return MockQueryStrategyType()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MockQueryStrategyType)

    def __hash__(self) -> int:
        return hash('mock')


@Stringifiable.make_stringifiable()
@dataclasses.dataclass(slots=True, frozen=True)
class ColdStartStrategy(Stringifiable):
    query_strategy: QueryStrategyType
    batch_size: int  # Размер выборки для обучения на одной итерации

    STR_PATTERN = re.compile(r'(?P<query_strategy>\w+)_(?P<budget>\d+)_(?P<batch_size>\d+)')

    @property  # TODO: make multiple iterations possible
    def budget(self) -> int:
        return self.batch_size

    @property
    def n_iterations(self) -> int:
        return 1

    def __str__(self) -> str:
        return f'{self.query_strategy.query_strategy_name()}_{self.budget}_{self.batch_size}'

    @staticmethod
    def from_str(strategy_str: str) -> 'ColdStartStrategy':
        m = re.fullmatch(ColdStartStrategy.STR_PATTERN, strategy_str)
        if m is None:
            raise ValueError('Invalid ColdStartStrategy string representation')
        query_strategy_str = m.group('query_strategy')
        budget = int(m.group('budget'))
        batch_size = int(m.group('batch_size'))
        if budget != batch_size:
            raise ValueError('Inconsistent budget and batch_size for ColdStartStrategy')
        return ColdStartStrategy(
            query_strategy=QueryStrategyType.factory_from_str(query_strategy_str),
            batch_size=batch_size,
        )

    @staticmethod
    def from_budget(query_strategy: QueryStrategyType, budget: int) -> 'ColdStartStrategy':
        return ColdStartStrategy(query_strategy=query_strategy, batch_size=budget)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColdStartStrategy):
            return False
        return (
            self.query_strategy.query_strategy_name() == other.query_strategy.query_strategy_name()
            and self.batch_size == other.batch_size
            and self.budget == other.budget
        )

    def __hash__(self):
        return hash((self.query_strategy.query_strategy_name(), self.batch_size, self.budget))

    def __repr__(self) -> str:
        return f'ColdStartStrategy(query_strategy={self.query_strategy}, batch_size={self.batch_size}, budget={self.budget}, n_iterations={self.n_iterations})'


@Stringifiable.make_stringifiable()
class ActiveLearningStrategy(Stringifiable):
    def __init__(
        self,
        query_strategy: QueryStrategyType,
        batch_size: int,  # Размер выборки для обучения на одной итерации
        budget: int,  # Общий бюджет на активное обучение
    ):
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.budget = budget

    STR_PATTERN = re.compile(r'(?P<query_strategy>\w+)_(?P<batch_size>\d+)_(?P<budget>\d+)')

    @property
    def n_iterations(self) -> int:
        return self.budget // self.batch_size + (1 if self.budget % self.batch_size > 0 else 0)

    def __str__(self) -> str:
        return f'{self.query_strategy.query_strategy_name()}_{self.batch_size}_{self.budget}'

    @staticmethod
    def from_str(strategy_str: str) -> 'ActiveLearningStrategy':
        m = re.fullmatch(ActiveLearningStrategy.STR_PATTERN, strategy_str)
        if m is None:
            raise ValueError('Invalid ActiveLearningStrategy string representation')
        query_strategy_str = m.group('query_strategy')
        batch_size = int(m.group('batch_size'))
        budget = int(m.group('budget'))
        if budget != batch_size * (budget // batch_size) + (budget % batch_size):
            raise ValueError('Inconsistent budget and batch_size for ActiveLearningStrategy')
        return ActiveLearningStrategy(
            query_strategy=QueryStrategyType.factory_from_str(query_strategy_str),
            batch_size=batch_size,
            budget=budget,
        )

    def batch_size_at(self, iteration: int) -> int:
        if iteration < 0 or iteration >= self.n_iterations:
            raise ValueError('Invalid iteration number')
        if iteration < self.n_iterations - 1:
            return self.batch_size
        else:
            return self.budget - self.batch_size * (self.n_iterations - 1)

    def __repr__(self):
        return f'ActiveLearningStrategy(query_strategy={self.query_strategy}, batch_size={self.batch_size}, budget={self.budget}, n_iterations={self.n_iterations})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActiveLearningStrategy):
            return False
        return (
            self.query_strategy.query_strategy_name() == other.query_strategy.query_strategy_name()
            and self.batch_size == other.batch_size
            and self.budget == other.budget
        )

    def __hash__(self):
        return hash((self.query_strategy.query_strategy_name(), self.batch_size, self.budget))


class ComposeStrategyWrapper(QueryStrategy):
    def __init__(self, al_strategy: ActiveLearningStrategy, cs_strategy: ColdStartStrategy, num_classes: int):
        self.al_strategy = al_strategy
        self.cs_strategy = cs_strategy
        self.num_classes = num_classes
        self.__budget_used = 0
        self.__al_strategy = None
        self.__cs_strategy = None

    @property
    def cold_start_strategy(self) -> QueryStrategy:
        if self.__cs_strategy is None:
            self.__cs_strategy = self.cs_strategy.query_strategy.query_strategy_class(self.num_classes)
        return self.__cs_strategy

    @property
    def active_learning_strategy(self) -> QueryStrategy:
        if self.__al_strategy is None:
            self.__al_strategy = self.al_strategy.query_strategy.query_strategy_class(self.num_classes)
        return self.__al_strategy

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        remaining_cs_budget = max(0, self.cs_strategy.budget - self.__budget_used)
        if remaining_cs_budget > 0:
            strategy = self.cold_start_strategy
        else:
            strategy = self.active_learning_strategy
        if remaining_cs_budget > 0 and n > remaining_cs_budget:
            raise ValueError('Requested more samples than available in cold start pool')
        self.__budget_used += n
        return strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

    def __repr__(self):
        return f'ComposeStrategyWrapper(phase={'"Cold start"' if self.__budget_used < self.cs_strategy.budget else '"Active learning"'}, budget_used={self.__budget_used}, al_strategy={self.al_strategy}, cs_strategy={self.cs_strategy}, num_classes={self.num_classes})'
