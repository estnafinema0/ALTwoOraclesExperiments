from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from small_text import PoolBasedActiveLearner
    from small_text.integrations.transformers.datasets import TransformersDataset

from utils import EnumABCMeta
import database
import llms
from storage import Stringifiable, Format

from small_text.query_strategies import QueryStrategy
from small_text.integrations.pytorch.query_strategies import BADGE as st_BADGE
from small_text.query_strategies.bayesian import BALD as st_BALD
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import scipy

import enum
from typing import Self, Any, Callable, Optional
import dataclasses
import datetime
from abc import abstractmethod
import time
import itertools
import functools
import re
import random


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
    def get_parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def query_strategy_name(self) -> str:
        pass

    @abstractmethod
    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__str__()})'

    @staticmethod
    @abstractmethod
    def from_str(s: str, db: 'database.DataDatabase') -> 'QueryStrategyType':
        pass

    @classmethod
    def factory_from_str(cls, s: str, db: 'database.DataDatabase') -> 'QueryStrategyType':
        for subclass in cls.__get_all_subclasses():
            try:
                return subclass.from_str(s, db)
            except ValueError:
                continue
        raise ValueError(f'Unknown QueryStrategyType string: {s}')

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    def set_pool(self, pool: 'database.Pool'):
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
    def from_str(
        cls: 'SimpleQueryStrategyType', strategy_str: str, db: 'database.DataDatabase'
    ) -> 'SimpleQueryStrategyType':
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

    def get_parameters(self) -> dict[str, Any]:
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
    def from_str(s: str, db: 'database.DataDatabase') -> 'QueryStrategySimple':
        return QueryStrategySimple(SimpleQueryStrategyType.from_str(s, db))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QueryStrategySimple) and self.__strategy == other.__strategy

    def __hash__(self) -> int:
        return hash(self.__strategy)


@Stringifiable.make_stringifiable()
class MockQueryStrategyType(QueryStrategyType, Stringifiable):
    def get_parameters(self) -> dict[str, Any]:
        return {}

    def query_strategy_name(self) -> str:
        return 'mock'

    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
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
    def from_str(s: str, db: 'database.DataDatabase') -> 'MockQueryStrategyType':
        if s != 'mock':
            raise ValueError('Invalid string for MockQueryStrategyType')
        return MockQueryStrategyType()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MockQueryStrategyType)

    def __hash__(self) -> int:
        return hash('mock')


class ActiveLLMRetriver:
    def __init__(
        self,
        chat: llms.Chat,
        get_duplicates_prompt: Callable[[npt.NDArray[np.uint], npt.NDArray[np.uint]], str],
        new_extraction_func: Callable[[int], Callable[[str], npt.NDArray[np.int64]]],
        get_unreadable_prompt: Callable[[int], str],
        get_out_of_range_prompt: Callable[[npt.NDArray[np.uint], int], str],
        llm: llms.LLM,
        actions_count: int,
    ):
        self.chat = chat
        self.get_duplicates_prompt = get_duplicates_prompt
        self.get_unreadable_prompt = get_unreadable_prompt
        self.get_out_of_range_prompt = get_out_of_range_prompt
        self.new_extraction_func = new_extraction_func
        self.actions_count = actions_count
        self.llm = llm

    def something_wrong(
        self, examples: npt.NDArray[llms.ExampleIndex], indices_labeled: npt.NDArray[np.uint], pool_size: int, n: int
    ):
        return (
            self.has_duplicates(examples, indices_labeled)
            or self.has_out_of_range(examples, pool_size)
            or self.has_too_few(examples, n)
        )

    def has_duplicates(self, examples: npt.NDArray[llms.ExampleIndex], indices_labeled: npt.NDArray[np.uint]) -> bool:
        return np.any(np.isin(examples, indices_labeled))

    def has_out_of_range(self, examples: npt.NDArray[llms.ExampleIndex], pool_size: int) -> bool:
        return np.any((examples < 0) | (examples >= pool_size))

    def has_too_few(self, examples: npt.NDArray[llms.ExampleIndex], n: int) -> bool:
        return examples.size < n

    def retrieve_examples(self, basic_prompt: str, n: int, indices_labeled: npt.NDArray[np.uint], pool_size: int):
        actions_left = self.actions_count
        try:
            examples = self.llm.query_examples(self.chat, basic_prompt, self.new_extraction_func(n))
        except TypeError:
            examples, actions_left = self.ask_refetch(actions_left, n)

        while self.something_wrong(examples, indices_labeled, pool_size, n):
            if self.has_too_few(examples, n):
                assert False
            elif self.has_duplicates(examples, indices_labeled):
                mask = np.where(np.isin(examples, indices_labeled))[0]
                fixed_duplicates, actions_left = self.fix_dupliactes_in_last_message(
                    actions_left, indices_labeled, examples, mask
                )
                examples[mask] = fixed_duplicates
            elif self.has_out_of_range(examples, pool_size):
                mask = np.where((examples < 0) | (examples >= pool_size))[0]
                fixed_out_of_range, actions_left = self.fix_out_of_range(actions_left, examples, mask, pool_size)
                examples[mask] = fixed_out_of_range

        assert not self.has_duplicates(examples, indices_labeled)

        return examples

    def fix_dupliactes_in_last_message(
        self,
        left_actions: int,
        indices_labeled: npt.NDArray[np.uint],
        last_fetched: npt.NDArray[llms.ExampleIndex],
        duplicates_mask: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[llms.ExampleIndex], int]:
        if not left_actions:
            raise ValueError('LLM Actions left!')

        duplicates = last_fetched[duplicates_mask]
        duplicates_prompt = self.get_duplicates_prompt(duplicates, indices_labeled)
        try:
            new_fetched = self.llm.query_examples(
                self.chat, duplicates_prompt, self.new_extraction_func(duplicates.size)
            )
        except TypeError:
            new_fetched, left_actions = self.ask_refetch(left_actions, duplicates.size)

        if not left_actions:
            raise ValueError('LLM Actions left!')

        if self.has_too_few(new_fetched, duplicates.size):
            assert False
        if self.has_duplicates(new_fetched, indices_labeled):
            mask = np.where(np.isin(new_fetched, indices_labeled))[0]
            new_fetched_fixed, left_actions = self.fix_dupliactes_in_last_message(
                left_actions - 1, indices_labeled, new_fetched, mask
            )
            new_fetched[mask] = new_fetched_fixed
            return new_fetched, left_actions - 1
        return new_fetched, left_actions - 1

    def ask_refetch(self, left_actions: int, n: int) -> tuple[npt.NDArray[llms.ExampleIndex], int]:
        if not left_actions:
            raise ValueError('LLM Actions left!')

        unreadable_prompt = self.get_unreadable_prompt(n)
        try:
            new_fetched = self.llm.query_examples(self.chat, unreadable_prompt, self.new_extraction_func(n))
        except TypeError:
            return self.ask_refetch(left_actions - 1, n)

        return new_fetched, left_actions - 1

    def fix_out_of_range(
        self,
        left_actions: int,
        last_fetched: npt.NDArray[llms.ExampleIndex],
        out_of_range_mask: npt.NDArray[np.intp],
        pool_size: int,
    ) -> tuple[npt.NDArray[llms.ExampleIndex], int]:
        if not left_actions:
            raise ValueError('LLM Actions left!')

        out_of_range = last_fetched[out_of_range_mask]
        out_of_range_prompt = self.get_out_of_range_prompt(out_of_range, pool_size)
        try:
            new_fetched = self.llm.query_examples(
                self.chat, out_of_range_prompt, self.new_extraction_func(out_of_range.size)
            )
        except TypeError:
            new_fetched, left_actions = self.ask_refetch(left_actions, out_of_range.size)

        if not left_actions:
            raise ValueError('LLM Actions left!')

        if self.has_too_few(new_fetched, out_of_range.size):
            assert False
        if self.has_out_of_range(new_fetched, pool_size):
            mask = np.where((new_fetched < 0) | (new_fetched >= pool_size))[0]
            new_fetched_fixed, left_actions = self.fix_out_of_range(left_actions - 1, new_fetched, mask, pool_size)
            new_fetched[mask] = new_fetched_fixed
            return new_fetched, left_actions - 1

        return new_fetched, left_actions - 1


@Stringifiable.make_stringifiable()
class ActiveLLMQueryStrategyType(QueryStrategyType, QueryStrategy):
    def __init__(self, llm_type: llms.LLMType, db: 'database.DataDatabase'):
        self.__llm_type = llm_type
        self.__llm = None
        self.__database = db
        assert self.__database.llms is not None
        self.__pool = None

    STR_PATTERN = re.compile(r'activellm__(?P<llm>[a-z\_A-Z0-9]+)')

    def set_pool(self, pool: 'database.Pool'):
        self.__pool = pool

    def set_db(self, db: 'database.DataDatabase'):
        self.__database = db

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def query_strategy_name(self) -> str:
        return str(self)

    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
        return self

    @property
    def llm(self) -> llms.LLM:
        if self.__llm is None:
            self.__llm = self.__database.get_llm(self.__llm_type)
        return self.__llm

    @staticmethod
    def __get_system_prompt(
        examples: npt.NDArray[np.str_],
        examples_count: int = 10,
    ) -> str:
        return f'''Представь себя в роли эксперта активного обучения, помогающего человеку-аннотатору. Твоя задача — выбрать примеры, которые аннотатор должен разметить. Тебе дан набор примеров. Ты можешь выбрать только {examples_count} примеров. В идеале следует выбрать те примеры, которые обеспечат модели наиболее информативные и разнообразные данные. Ниже приведены стратегии, на которые стоит ориентироваться:

Репрезентативность (Representativeness):
Выбирай примеры, которые репрезентативны для всего датасета. Это гарантирует, что модель увидит разнообразные примеры и сможет лучше обобщать.

Разнообразие (Diversity):
В рамках критерия репрезентативности выбирай примеры, охватывающие широкий спектр сценариев. Это могут быть граничные случаи или редкие ситуации.

Сложность или неопределённость (Difficulty or Uncertainty):
Сложные для модели примеры особенно ценны, поскольку помогают улучшить её работу в слабых местах.

Стратифицированная выборка (Stratified Sampling):
Если данные можно разбить на страты (например, категории или диапазоны непрерывной переменной), убедись, что размеченные примеры включают представителей из каждой страты.

Баланс классов (Balancing Classes):
Следи за тем, чтобы в размеченном наборе не было чрезмерного перекоса в сторону более распространённых классов.

Избегание предвзятости (Avoid Bias):
Следи за тем, чтобы твой выбор не вносил предвзятость. Например, избегай стереотипов или избыточного представления отдельных групп.

Формат вывода:
Выход должен представлять собой список выбранных примеры, разделённых запятыми.
Например, если ты выбираешь примеры 3, 42, 666, то вывод должен быть:
`3, 42, 666`

Ниже приведены примеры (разделены с помощью "\n\n ##### \n\n"):

{"\n\n ##### \n\n".join(f'{idx}: {example}' for idx, example in zip(np.arange(examples.size), examples))}

Ты можешь выбрать только {examples_count} примеров.'''

    @staticmethod
    def __get_basic_message_prompt(
        labeled: npt.NDArray[np.uint],
    ) -> str:
        return (
            f"""
Следующие примеры уже были размечены (индексы из датасета выше), их добавлять не надо заново для разметки:

{', '.join(map(str, labeled))}"""
            if labeled.size != 0
            else ''
        )

    @staticmethod
    def __get_duplicates_prompt(
        examples: npt.NDArray[np.uint],
        labeled: npt.NDArray[np.uint],
    ) -> str:
        return f'''Эти примеры уже были размечены, их надо заменить на те, которых ещё не было, выбери ровно {examples.size} других примеров:
{', '.join(map(str, map(int, examples)))}
Выведи только новые примеры списком индексов, выводить что было НЕ НУЖНО, выводить, как они будут включены НЕ НУЖНО, выведи например так:
`{random.sample(range(1000), examples.size)}`'''

    @staticmethod
    def __get_unreadable_prompt(n: int) -> str:
        return f'''Твои примеры были не в запрошенном формате. Я не смог их прочитать. Выведи ещё раз, например так
`{random.sample(range(1000), n)}`'''

    @staticmethod
    def __get_out_of_range_prompt(examples: npt.NDArray[np.uint], pool_size: int) -> str:
        return f'''Эти примеры вышли за диапазон, их не существует:
{', '.join(map(str, map(int, examples)))}
Поменяй их на примеры от 0 до {pool_size - 1}'''

    def __new_extraction_func(self, examples_count: int) -> Callable[[str], npt.NDArray[np.int64]]:
        DIGITS = re.compile(
            r'(?P<digits>\*{0,3}\d+\*{0,3}(,[\s\xa0]*\*{0,3}\d+\*{0,3}){' + str(examples_count - 1) + r',}?)'
        )
        self.__database.logger.debug(f'{DIGITS=}')

        def __extract_from_query(s: str) -> npt.NDArray[np.int64]:
            self.__database.logger.debug(f'{s=}')

            def clear_stars(num: str) -> str:
                return num.strip('*')

            SEP_PATTERN = r',[\s\xa0]*'

            last_digits = tuple(
                map(
                    int,
                    map(
                        clear_stars,
                        re.split(SEP_PATTERN, functools.reduce(lambda _, x: x, re.finditer(DIGITS, s)).group('digits')),
                    ),
                )
            )[:examples_count]
            return np.array(last_digits, dtype=np.int64)

        return __extract_from_query

    def __chat_id(self) -> str:
        return self.__pool.get_id()

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled: npt.NDArray[np.uint],
        indices_labeled: npt.NDArray[np.uint],
        y: npt.NDArray[np.uint],
        n=10,
    ):
        llm_recap_id = database.LLMIndexRecapExamples.make_id(
            self.__pool.base.id, self.__pool.subset, self.__llm_type, n
        )
        llm_recap = (
            self.__database.retrieve(llm_recap_id)
            if llm_recap_id in self.__database
            else database.LLMIndexRecapExamples(self.__pool.base.id, self.__pool.subset, self.__llm_type, {}, n)
        )

        bunch = frozenset(indices_labeled)
        if bunch in llm_recap.recapped:
            return llm_recap.recapped[bunch]

        retriver = ActiveLLMRetriver(
            self.llm.make_chat(self.__get_system_prompt(self.__pool.x, examples_count=n), self.__chat_id()),
            self.__get_duplicates_prompt,
            self.__new_extraction_func,
            self.__get_unreadable_prompt,
            self.__get_out_of_range_prompt,
            self.llm,
            actions_count=10,
        )
        examples = retriver.retrieve_examples(
            self.__get_basic_message_prompt(indices_labeled), n, indices_labeled, self.__pool.size
        )

        llm_recap.recapped[bunch] = examples
        self.__database.store_fast(llm_recap.as_storable(), Format.JSON)
        return examples

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
        return f'activellm__{str(self.__llm_type)}'

    @staticmethod
    def from_str(s: str, db: 'database.DataDatabase') -> 'ActiveLLMQueryStrategyType':
        m = re.fullmatch(ActiveLLMQueryStrategyType.STR_PATTERN, s)
        if m is None:
            raise ValueError(f'Invalid ActiveLearningStrategy string representation: {s}')
        llm_type = llms.LLMType(m.group('llm'))
        return ActiveLLMQueryStrategyType(llm_type, db)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActiveLLMQueryStrategyType):
            return False
        return self.llm == other.llm and self.__pool == other.__pool

    def __hash__(self) -> int:
        return hash('ActiveLLMQueryStrategyType')


class SelectLLMRetriver:
    def __init__(
        self,
        chat: llms.Chat,
        new_extraction_func: Callable[[], Callable[[str], np.int64]],
        get_unreadable_prompt: Callable[[int], str],
        get_out_of_cluster_prompt: Callable[[np.uint, int], str],
        llm: llms.LLM,
        actions_count: int,
    ):
        self.chat = chat
        self.get_unreadable_prompt = get_unreadable_prompt
        self.get_out_of_cluster_prompt = get_out_of_cluster_prompt
        self.new_extraction_func = new_extraction_func
        self.actions_count = actions_count
        self.llm = llm

    def something_wrong(self, example: int, cluster_size: int):
        return self.is_out_of_cluster(example, cluster_size)

    def is_out_of_cluster(self, example: int, cluster_size: int) -> bool:
        return example < 1 or example > cluster_size

    def retrieve_examples(self, basic_prompt: str, cluster_size: int) -> np.int64:
        actions_left = self.actions_count
        try:
            example = self.llm.query_examples(self.chat, basic_prompt, self.new_extraction_func())
        except TypeError:
            example, actions_left = self.ask_refetch(actions_left, cluster_size)

        while self.something_wrong(example, cluster_size):
            if self.is_out_of_cluster(example, cluster_size):
                example, actions_left = self.fix_out_of_cluster(actions_left, example, cluster_size)

        return np.int64(example)

    def ask_refetch(self, left_actions: int, n: int) -> tuple[np.int64, int]:
        if not left_actions:
            raise ValueError('LLM Actions left!')

        unreadable_prompt = self.get_unreadable_prompt(n)
        try:
            new_fetched = self.llm.query_examples(self.chat, unreadable_prompt, self.new_extraction_func())
        except TypeError:
            return self.ask_refetch(left_actions - 1, n)

        return new_fetched, left_actions - 1

    def fix_out_of_cluster(
        self,
        left_actions: int,
        example: np.int64,
        cluster_size: int,
    ) -> tuple[np.int64, int]:
        if not left_actions:
            raise ValueError('LLM Actions left!')

        out_of_cluster_prompt = self.get_out_of_cluster_prompt(example, cluster_size)
        try:
            new_fetched = self.llm.query_examples(self.chat, out_of_cluster_prompt, self.new_extraction_func())
        except TypeError:
            new_fetched, left_actions = self.ask_refetch(left_actions, cluster_size)

        if not left_actions:
            raise ValueError('LLM Actions left!')

        if self.is_out_of_cluster(new_fetched, cluster_size):
            new_fetched, left_actions = self.fix_out_of_cluster(left_actions - 1, new_fetched, cluster_size)
            return new_fetched, left_actions - 1

        return new_fetched, left_actions - 1


@Stringifiable.make_stringifiable()
class SelectLLMQueryStrategyType(QueryStrategyType, QueryStrategy):
    def __init__(self, llm_type: llms.LLMType, db: 'database.DataDatabase'):
        self.__llm_type = llm_type
        self.__llm = None
        self.__database = db
        assert self.__database.llms is not None
        self.__pool = None
        self.__embed_model = None

    STR_PATTERN = re.compile(r'selectllm__(?P<llm>[a-z\_A-Z0-9]+)')

    def set_pool(self, pool: 'database.Pool'):
        self.__pool = pool

    def set_db(self, db: 'database.DataDatabase'):
        self.__database = db

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def query_strategy_name(self) -> str:
        return str(self)

    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
        return self

    @property
    def llm(self) -> llms.LLM:
        if self.__llm is None:
            self.__llm = self.__database.get_llm(self.__llm_type)
        return self.__llm

    @property
    def embed_model(self) -> SentenceTransformer:
        if self.__embed_model is None:
            self.__embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.__embed_model

    @staticmethod
    def __get_system_prompt(examples_count: int) -> str:
        return f'''Ниже приведены {examples_count} неразмеченных предложений. Каждое предложение обозначено числовым идентификатором в квадратных скобках [].

Выбери одно предложение, которое наиболее полезно разметить человеку для последующего обучения модели классификации текста.

Предпочитай предложение, которое:
- содержательно и понятно;
- потенциально сложно для классификации;
- отражает важный или нетривиальный случай;
- отличается по формулировке, контексту и смыслу;
- может дать наибольший прирост качества модели после разметки.

Верни только идентификатор выбранного предложения в формате [].
Например: [1] или [666].
'''

    @staticmethod
    def __get_basic_message_prompt(examples: npt.NDArray[np.str_]) -> str:
        return f'''{'\n'.join(f"[{i}]\n### Текст: {example}\n" for i, example in enumerate(examples, 1))}
Наиболее полезное предложение (только идентификатор):'''

    @staticmethod
    def __get_unreadable_prompt(size: int) -> str:
        return f'''Твой пример был не в запрошенном формате. Я не смог его прочитать. Выведи ещё раз, например так:
[{random.choice(range(1, size + 1))}]'''

    @staticmethod
    def __get_out_of_cluster_prompt(example: np.uint, size: int) -> str:
        return f'''Этот пример вышел за диапазон, его не существует: [{example}]
Поменяй его на пример от 1 до {size}'''

    def __new_extraction_func(self) -> Callable[[str], np.int64]:
        CLUSTER = re.compile(r'(?P<cluster_number>\[\d+\])')

        def __extract_from_query(s: str) -> npt.NDArray[np.int64]:
            def clear_brackets(s: str) -> str:
                return s.lstrip('[').rstrip(']')

            last_cluster_number = int(
                clear_brackets(functools.reduce(lambda _, x: x, re.finditer(CLUSTER, s)).group('cluster_number'))
            )
            return np.int64(last_cluster_number)

        return __extract_from_query

    def __chat_id(self) -> str:
        return self.__pool.get_id()

    def __embed_pool(self, pool: 'database.Pool') -> npt.NDArray[np.floating]:
        return self.embed_model.encode([f"### Текст: {x}" for x in pool.x])

    def __build_diverse_kmeans_clusters(self, n: int) -> tuple[npt.NDArray[np.int64], ...]:
        train_embeddings = self.__embed_pool(self.__pool)
        n_samples = len(train_embeddings)

        min_group_size = n_samples // n
        extra_groups = n_samples % n
        max_group_size = min_group_size + (extra_groups > 0)

        final_group_sizes = np.array(
            [max_group_size] * extra_groups + [min_group_size] * (n - extra_groups), dtype=np.int64
        )

        # transpose viewpoint:
        # row/center 0 contributes one item to all n groups
        # ...
        # row/center min_group_size-1 contributes one item to all n groups
        # optional last row contributes one item only to extra_groups groups
        center_capacities = np.array(
            [n] * min_group_size + ([extra_groups] if extra_groups > 0 else []), dtype=np.int64
        )

        n_centers = len(center_capacities)
        assert center_capacities.sum() == n_samples

        kmeans = KMeans(n_clusters=n_centers, n_init="auto").fit(train_embeddings)

        dist_to_centers = kmeans.transform(train_embeddings)  # shape: (n_samples, n_centers)

        assigned_center = -1 * np.ones(n_samples, dtype=np.int64)
        available = np.ones(n_samples, dtype=bool)

        # exact-capacity greedy assignment, no duplication
        for center_id in range(n_centers):
            for _ in range(int(center_capacities[center_id])):
                available_indices = np.nonzero(available)[0]
                if len(available_indices) == 0:
                    raise ValueError('No examples left to assign')

                local_best = np.argmin(dist_to_centers[available_indices, center_id])
                selected_index = int(available_indices[local_best])

                assigned_center[selected_index] = center_id
                available[selected_index] = False

        if np.any(assigned_center == -1):
            raise ValueError('Unassigned examples remain')

        # collect examples per intermediate center
        by_center: list[list[int]] = [[] for _ in range(n_centers)]
        for idx, center_id in enumerate(assigned_center):
            by_center[int(center_id)].append(idx)

        res: list[list[int]] = [[] for _ in range(n)]
        for center_id in range(n_centers):
            capacity = int(center_capacities[center_id])
            for p in range(capacity):
                res[p].append(by_center[center_id][p])

        for group_id in range(n):
            if len(res[group_id]) != int(final_group_sizes[group_id]):
                raise ValueError(
                    f'Expected group {group_id} size {final_group_sizes[group_id]}, ' f'got {len(res[group_id])}'
                )

        # strict no-overlap check
        flat = [idx for group in res for idx in group]
        if len(flat) != len(set(flat)):
            raise ValueError('Clusters overlap, which should be impossible here')

        return tuple(np.array(group, dtype=np.int64) for group in res)

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled: npt.NDArray[np.uint],
        indices_labeled: npt.NDArray[np.uint],
        y: npt.NDArray[np.uint],
        n=10,
    ):
        clusterss_id = database.DiversityBasedKMeansClusters.make_id(self.__pool.base.id, self.__pool.subset)
        clusterss = (
            self.__database.retrieve(clusterss_id)
            if clusterss_id in self.__database
            else database.DiversityBasedKMeansClusters(self.__pool.base.id, self.__pool.subset, {})
        )

        if n in clusterss.clustered:
            clusters = clusterss.clustered[n]
        else:
            clusters = self.__build_diverse_kmeans_clusters(n)
            clusterss.clustered[n] = clusters
            self.__database.store_fast(clusterss.as_storable(), Format.JSON)

        cluster_examples_id = database.LLMClusterExamples.make_id(
            self.__pool.base.id, self.__llm_type, self.__pool.subset
        )
        cluster_exampless = (
            self.__database.retrieve(cluster_examples_id)
            if cluster_examples_id in self.__database
            else database.LLMClusterExamples(self.__pool.base.id, self.__pool.subset, self.__llm_type, {})
        )

        examples = []
        for cluster in clusters:
            bunch = frozenset(map(int, cluster))
            if bunch in cluster_exampless.cluster_to_examples:
                example = cluster_exampless.cluster_to_examples[bunch]
            else:
                retriver = SelectLLMRetriver(
                    self.llm.make_chat(self.__get_system_prompt(examples_count=cluster.size), self.__chat_id()),
                    self.__new_extraction_func,
                    self.__get_unreadable_prompt,
                    self.__get_out_of_cluster_prompt,
                    self.llm,
                    actions_count=10,
                )
                cluster_examples = self.__pool.x[cluster]
                example = retriver.retrieve_examples(self.__get_basic_message_prompt(cluster_examples), cluster.size)
                example = cluster[example - 1]
                cluster_exampless.cluster_to_examples[bunch] = int(example)
                self.__database.store_fast(cluster_exampless.as_storable(), Format.JSON)
            examples.append(example)

        return np.array(examples, dtype=np.int64)

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
        return f'selectllm__{str(self.__llm_type)}'

    @staticmethod
    def from_str(s: str, db: 'database.DataDatabase') -> 'SelectLLMQueryStrategyType':
        m = re.fullmatch(SelectLLMQueryStrategyType.STR_PATTERN, s)
        if m is None:
            raise ValueError(f'Invalid ActiveLearningStrategy string representation: {s}')
        llm_type = llms.LLMType(m.group('llm'))
        return SelectLLMQueryStrategyType(llm_type, db)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectLLMQueryStrategyType):
            return False
        return self.llm == other.llm and self.__pool == other.__pool

    def __hash__(self) -> int:
        return hash('SelectLLMQueryStrategyType')


@Stringifiable.make_stringifiable()
class DiversityInitQueryStrategyType(QueryStrategyType, QueryStrategy):
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        token_percentage: float = 0.15,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.__model_name = model_name
        self.__max_length = max_length
        self.__token_percentage = token_percentage
        self.__device = device
        self.__batch_size = batch_size

        self.__pool = None
        self.__tokenizer = None
        self.__mlm_model = None

    STR_PATTERN = re.compile(
        r'diversityinit__(?P<model>[\w\-/\.]+)__(?P<max_length>\d+)__(?P<token_percentage>\d+(?:\.\d+)?)'
    )

    def set_pool(self, pool: 'database.Pool'):
        self.__pool = pool

    def get_parameters(self) -> dict[str, Any]:
        return {
            'model_name': self.__model_name,
            'max_length': self.__max_length,
            'token_percentage': self.__token_percentage,
            'device': self.device,
            'batch_size': self.__batch_size,
        }

    def query_strategy_name(self) -> str:
        return str(self)

    def query_strategy_class(self, num_classes: int) -> QueryStrategy:
        return self

    @property
    def device(self) -> str:
        if self.__device is None:
            self.__device = 'mps' if torch.mps.is_available() else 'cpu'
        return self.__device

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self.__tokenizer is None:
            self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        return self.__tokenizer

    @property
    def mlm_model(self) -> AutoModelForMaskedLM:
        if self.__mlm_model is None:
            self.__mlm_model = AutoModelForMaskedLM.from_pretrained(self.__model_name)
            self.__mlm_model.to(self.device)
            self.__mlm_model.eval()
        return self.__mlm_model

    def __build_masked_lm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.__token_percentage)

        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True) for row in labels],
            dtype=torch.bool,
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if self.tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        return labels

    def __surprisal_embeddings(self, indices_unlabeled: npt.NDArray[np.int64]) -> npt.NDArray[np.floating]:
        texts = list(map(str, self.__pool.x[indices_unlabeled]))
        all_losses: list[np.ndarray] = []

        for start in range(0, len(texts), self.__batch_size):
            batch_texts = texts[start : start + self.__batch_size]

            enc = self.tokenizer(
                batch_texts, padding='max_length', truncation=True, max_length=self.__max_length, return_tensors='pt'
            )

            input_ids_cpu = enc['input_ids'].clone()
            labels_cpu = self.__build_masked_lm_labels(input_ids_cpu)

            input_ids = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)
            labels = labels_cpu.to(self.device)

            with torch.no_grad():
                logits = self.mlm_model(input_ids=input_ids, attention_mask=attention_mask).logits

            batch_size, seq_length, vocab_size = logits.shape
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
            loss = loss_batched.view(batch_size, seq_length)

            all_losses.append(loss.detach().cpu().numpy())

        surprisal = np.concatenate(all_losses, axis=0)
        return surprisal

    def __pick_representatives_like_repo(
        self,
        embeddings: npt.NDArray[np.floating],
        centers: npt.NDArray[np.floating],
        global_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        centroids = np.unique(scipy.spatial.distance.cdist(centers, embeddings).argmin(axis=1))

        missing = centers.shape[0] - len(centroids)
        if missing > 0:
            pool = np.delete(np.arange(len(embeddings)), centroids)
            extra_local = np.random.choice(pool, size=missing, replace=False)
            chosen_local = np.concatenate((centroids, extra_local), axis=None)
        else:
            chosen_local = centroids

        return global_indices[np.asarray(chosen_local, dtype=np.int64)]

    def query(
        self,
        clf,
        dataset,
        indices_unlabeled: npt.NDArray[np.uint],
        indices_labeled: npt.NDArray[np.uint],
        y: npt.NDArray[np.uint],
        n: int = 10,
    ) -> npt.NDArray[np.int64]:
        if self.__pool is None:
            raise ValueError('Pool is not set')

        unlabeled = np.asarray(indices_unlabeled, dtype=np.int64)

        surprisal = self.__surprisal_embeddings(unlabeled)
        vectors = normalize(surprisal)

        kmeans = KMeans(n_clusters=n).fit(vectors)

        return self.__pick_representatives_like_repo(
            embeddings=vectors, centers=kmeans.cluster_centers_, global_indices=unlabeled
        )

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
            y_queried = pool.y[queried_indices]
            active_learner.update(y_queried)
            indices_labeled = np.concatenate([indices_labeled, queried_indices])

        end = time.perf_counter()

        acc, macro_f1 = evaluate_on_test(active_learner, test_dataset)
        duration = datetime.timedelta(seconds=end - start)
        return acc, macro_f1, duration, indices_labeled

    def __str__(self) -> str:
        return f'diversityinit__{self.__model_name}__{self.__max_length}__{self.__token_percentage}'

    @staticmethod
    def from_str(s: str, db: 'database.DataDatabase') -> 'DiversityInitQueryStrategyType':
        m = re.fullmatch(DiversityInitQueryStrategyType.STR_PATTERN, s)
        if m is None:
            raise ValueError(f'Invalid DiversityInitQueryStrategyType string representation: {s}')

        return DiversityInitQueryStrategyType(
            model_name=m.group('model'),
            max_length=int(m.group('max_length')),
            token_percentage=float(m.group('token_percentage')),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiversityInitQueryStrategyType):
            return False
        return (
            self.__model_name == other.__model_name
            and self.__max_length == other.__max_length
            and self.__token_percentage == other.__token_percentage
            and self.__pool == other.__pool
        )

    def __hash__(self) -> int:
        return hash(('DiversityInitQueryStrategyType', self.__model_name, self.__max_length, self.__token_percentage))


@Stringifiable.make_stringifiable()
@dataclasses.dataclass(slots=True, init=False)
class ColdStartStrategy(Stringifiable):
    query_strategy: QueryStrategyType
    batch_size: int  # Размер выборки для обучения на одной итерации
    budget: int

    def __init__(self, query_strategy: QueryStrategyType, batch_size: int, budget: Optional[int] = None):
        if budget is None:
            budget = batch_size

        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.budget = budget

    STR_PATTERN = re.compile(r'(?P<query_strategy>[\w_\d\-\.]+)_(?P<budget>\d+)_(?P<batch_size>\d+)')

    @property
    def n_iterations(self) -> int:
        return self.budget // self.batch_size + (1 if self.budget % self.batch_size > 0 else 0)

    def __str__(self) -> str:
        return f'{self.query_strategy.query_strategy_name()}_{self.budget}_{self.batch_size}'

    @staticmethod
    def from_str(strategy_str: str, db: 'database.DataDatabase') -> 'ColdStartStrategy':
        m = re.fullmatch(ColdStartStrategy.STR_PATTERN, strategy_str)
        if m is None:
            raise ValueError(f'Invalid ColdStartStrategy string representation: {strategy_str}')
        query_strategy_str = m.group('query_strategy')
        budget = int(m.group('budget'))
        batch_size = int(m.group('batch_size'))
        return ColdStartStrategy(
            query_strategy=QueryStrategyType.factory_from_str(query_strategy_str, db),
            batch_size=batch_size,
            budget=budget,
        )

    @staticmethod
    def from_budget(query_strategy: QueryStrategyType, budget: int) -> 'ColdStartStrategy':
        return ColdStartStrategy(query_strategy=query_strategy, batch_size=budget)

    def batch_size_at(self, iteration: int) -> int:
        if iteration < 0 or iteration >= self.n_iterations:
            raise ValueError('Invalid iteration number')
        if iteration < self.n_iterations - 1:
            return self.batch_size
        else:
            return self.budget - self.batch_size * (self.n_iterations - 1)

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

    STR_PATTERN = re.compile(r'(?P<query_strategy>[\w_\d]+)_(?P<batch_size>\d+)_(?P<budget>\d+)')

    @property
    def n_iterations(self) -> int:
        return self.budget // self.batch_size + (1 if self.budget % self.batch_size > 0 else 0)

    def __str__(self) -> str:
        return f'{self.query_strategy.query_strategy_name()}_{self.batch_size}_{self.budget}'

    @staticmethod
    def from_str(strategy_str: str, db: 'database.DataDatabase') -> 'ActiveLearningStrategy':
        m = re.fullmatch(ActiveLearningStrategy.STR_PATTERN, strategy_str)
        if m is None:
            raise ValueError('Invalid ActiveLearningStrategy string representation')
        query_strategy_str = m.group('query_strategy')
        batch_size = int(m.group('batch_size'))
        budget = int(m.group('budget'))
        if budget != batch_size * (budget // batch_size) + (budget % batch_size):
            raise ValueError('Inconsistent budget and batch_size for ActiveLearningStrategy')
        return ActiveLearningStrategy(
            query_strategy=QueryStrategyType.factory_from_str(query_strategy_str, db),
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
            if self.__budget_used == 0:
                print(f'[INFO]: {self.cs_strategy.batch_size}/{self.cs_strategy.budget}', flush=True, end='')
                if self.cs_strategy.budget - self.__budget_used - self.cs_strategy.batch_size == 0: print()
            elif self.cs_strategy.budget - self.__budget_used - self.cs_strategy.batch_size != 0:
                print(
                    f', {self.__budget_used + self.cs_strategy.batch_size}/{self.cs_strategy.budget}',
                    flush=True,
                    end='',
                )
            else:
                print(f', {self.__budget_used + self.cs_strategy.batch_size}/{self.cs_strategy.budget}', flush=True)
        else:
            strategy = self.active_learning_strategy

            if self.__budget_used == self.cs_strategy.budget:
                print(f'[INFO]: {self.al_strategy.batch_size}/{self.al_strategy.budget}', flush=True, end='')
                if self.__budget_used - self.cs_strategy.budget + self.al_strategy.batch_size == self.al_strategy.budget: print()
            elif self.__budget_used - self.cs_strategy.budget + self.al_strategy.batch_size != self.al_strategy.budget:
                print(
                    f', {self.__budget_used - self.cs_strategy.budget + self.al_strategy.batch_size}/{self.al_strategy.budget}',
                    flush=True,
                    end='',
                )
            else:
                print(
                    f', {self.__budget_used - self.cs_strategy.budget + self.al_strategy.batch_size}/{self.al_strategy.budget}',
                    flush=True,
                )
        if remaining_cs_budget > 0 and n > remaining_cs_budget:
            raise ValueError('Requested more samples than available in cold start pool')
        self.__budget_used += n
        return strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

    def __repr__(self):
        return f'ComposeStrategyWrapper(phase={'"Cold start"' if self.__budget_used < self.cs_strategy.budget else '"Active learning"'}, budget_used={self.__budget_used}, al_strategy={self.al_strategy}, cs_strategy={self.cs_strategy}, num_classes={self.num_classes})'
