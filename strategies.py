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

    def to_str(self) -> str:
        return self.STR_MAPPINGS[self]
    
    @classmethod
    def from_str(cls: 'SimpleQueryStrategyType', strategy_str: str) -> 'SimpleQueryStrategyType':
        return cls.REVERSE_STR_MAPPINGS[strategy_str]
        
class QueryStrategySimple(QueryStrategyType):
    def __init__(self, simple_strategy: SimpleQueryStrategyType):
        super().__init__()
        self.__strategy = simple_strategy

    def get_parameters(self) -> dict[str, any]:
        return {}
    
    def query_strategy_name(self) -> str:
        return self.__strategy.to_str()


@dataclasses.dataclass
class ColdStartStrategy:
    query_strategy: QueryStrategyType
    pool_size: int # Количество примеров, из которых берется начальный набор
    batch_size: int # Размер выборки для обучения на одной итерации
    
    @property # TODO: make multiple iterations possible
    def budget(self) -> int:
        return self.batch_size
    
@dataclasses.dataclass
class ActiveLearningStrategy:
    query_strategy: QueryStrategyType
    step_size: int # Размер выборки для обучения на одной итерации
    budget: int # Общий бюджет на активное обучение
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