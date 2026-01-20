import small_text
from small_text.query_strategies.base import QueryStrategy
from small_text.integrations.pytorch.query_strategies import BADGE
from small_text.query_strategies.bayesian import BALD
from small_text.query_strategies.strategies import RandomSampling, LeastConfidence
from small_text import TransformerBasedClassificationFactory, TransformerModelArguments

import enum
import dataclasses
from abc import ABC, abstractmethod

class QueryStrategyType(ABC):
    @abstractmethod
    def get_parameters(self) -> dict[str, any]:
        pass
    
    @abstractmethod
    def query_strategy_name(self) -> str:
        pass
    
    @abstractmethod
    def query_strategy_class(self) -> QueryStrategy:
        pass
        
class SimpleQueryStrategyType(enum.Enum):
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

    def to_str(self) -> str:
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


@dataclasses.dataclass
class ColdStartStrategy:
    query_strategy: QueryStrategyType
    pool_size: int # Количество примеров, из которых берется начальный набор
    batch_size: int # Размер выборки для обучения на одной итерации
    
    @property # TODO: make multiple iterations possible
    def budget(self) -> int:
        return self.batch_size

def make_classifier_factory(num_classes: int, transformer_model_name: str, device: str, num_epochs: int = 3, train_batch_size: int = 128) -> TransformerBasedClassificationFactory:
    model_args = TransformerModelArguments(transformer_model_name)
    
    ...

    clf_factory = TransformerBasedClassificationFactory(
        model_args,
        num_classes,
        kwargs=dict(
            device=device,
            num_epochs=num_epochs,
            mini_batch_size=train_batch_size,
        ),
    )
    return clf_factory


class ActiveLearningStrategy:
    # query_strategy: QueryStrategyType
    # step_size: int # Размер выборки для обучения на одной итерации
    # budget: int # Общий бюджет на активное обучение
    def __init__(
        self,
        query_strategy: QueryStrategyType,
        step_size: int, # Размер выборки для обучения на одной итерации
        budget: int, # Общий бюджет на активное обучение
    ):
        self.query_strategy = query_strategy
        self.step_size = step_size
        self.budget = budget

    @property
    def n_iterations(self) -> int:
        return self.budget // self.step_size + (1 if self.budget % self.step_size > 0 else 0)

    def batch_size_at(self, iteration: int) -> int:
        if iteration < 0 or iteration >= self.n_iterations:
            raise ValueError("Invalid iteration number")
        if iteration < self.n_iterations - 1:
            return self.step_size
        else:
            return self.budget - self.step_size * (self.n_iterations - 1)
        
    