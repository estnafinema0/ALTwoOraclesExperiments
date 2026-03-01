from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

import strategies
import database
from storage import Storable, StorableBundle, StorableEntry, StorableType, ID, Hash, Format
from utils import carried_partial_apply

from small_text import (
    TransformerModelArguments,
    TransformerBasedClassificationFactory,
)
import numpy as np

import datetime
import uuid
import builtins
import dataclasses
import itertools
import functools
from typing import Self, Optional, Callable, Iterable


@Storable.make_storable(StorableType.EXPERIMENT_HISTORY)
@dataclasses.dataclass(slots=True, frozen=True)
class ExperimentHistory(Storable):
    final_accuracy: float
    final_macro_f1: float
    after_cold_start_accuracy: float
    after_cold_start_macro_f1: float
    dataset_id: 'database.DatasetID'
    pool_size: int
    seed: int
    cs_strategy: strategies.ColdStartStrategy
    al_strategy: strategies.ActiveLearningStrategy
    duration_cs: datetime.timedelta
    duration_total: datetime.timedelta
    uuid: Optional[str] = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    format_version: int = 1

    @property
    def split(self) -> int:
        return self.cs_strategy.budget

    @property
    def budget(self) -> int:
        return self.cs_strategy.budget + self.al_strategy.budget

    @staticmethod
    def _get_salt(
        dataset_id: 'database.DatasetID',
        seed: int,
        cs_strategy: strategies.ColdStartStrategy,
        al_strategy: strategies.ActiveLearningStrategy,
    ) -> Hash:
        return Storable.combine_hashes(
            *map(
                Storable.hash_str,
                map(str, (dataset_id, seed, cs_strategy, al_strategy)),  # TODO: strategy hashing
            )
        )

    def get_id(self) -> ID:
        return f'{self.uuid}#{self._get_salt(self.dataset_id, self.seed, self.cs_strategy, self.al_strategy)}'

    @staticmethod
    def make_id(
        dataset_id: 'database.DatasetID',
        seed: int,
        cs_strategy: strategies.QueryStrategyType,
        al_strategy: strategies.QueryStrategyType,
    ) -> ID:
        raise ValueError('ExperimentHistory IDs should be generated automatically, make_id should not be used directly')

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        'final_accuracy': self.final_accuracy,
                        'final_macro_f1': self.final_macro_f1,
                        'after_cold_start_accuracy': self.after_cold_start_accuracy,
                        'after_cold_start_macro_f1': self.after_cold_start_macro_f1,
                        'dataset_id': str(self.dataset_id),
                        'pool_size': self.pool_size,
                        'seed': self.seed,
                        'cs_strategy': str(self.cs_strategy),
                        'al_strategy': str(self.al_strategy),
                        'format_version': self.format_version,
                        'uuid': self.uuid,
                        'duration_cs': self.duration_cs / datetime.timedelta(milliseconds=1),
                        'duration_total': self.duration_total / datetime.timedelta(milliseconds=1),
                    },
                    type=StorableType.EXPERIMENT_HISTORY,
                    id=self.get_id(),
                )
            },
        )

    @staticmethod
    def from_storable(
        entry: StorableEntry, data: 'database.DataDatabase', storable_type: 'StorableType | None' = None
    ) -> Self:
        if entry.id in data and storable_type != StorableType.EXPERIMENT_HISTORY:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = ExperimentHistory(
            final_accuracy=payload['final_accuracy'],
            final_macro_f1=payload['final_macro_f1'],
            after_cold_start_accuracy=payload['after_cold_start_accuracy'],
            after_cold_start_macro_f1=payload['after_cold_start_macro_f1'],
            dataset_id=database.DatasetID.from_str(payload['dataset_id']),
            pool_size=payload['pool_size'],
            seed=payload['seed'],
            cs_strategy=strategies.ColdStartStrategy.from_str(payload['cs_strategy']),
            al_strategy=strategies.ActiveLearningStrategy.from_str(payload['al_strategy']),
            format_version=payload['format_version'],
            uuid=payload['uuid'],
            duration_cs=datetime.timedelta(milliseconds=payload['duration_cs']),
            duration_total=datetime.timedelta(milliseconds=payload['duration_total']),
        )
        data.encache(obj)
        return obj


def make_classifier_factory(
    transformer_model_name: str, num_classes: int, train_batch_size: int, num_epochs: int = 3
) -> TransformerBasedClassificationFactory:
    import torch

    model_args = TransformerModelArguments(transformer_model_name)

    clf_factory = TransformerBasedClassificationFactory(
        model_args,
        num_classes,
        classification_kwargs=dict(
            device='cpu',  # "mps" if torch.mps.is_available() else "cpu",
            num_epochs=num_epochs,
            mini_batch_size=train_batch_size,
        ),
    )
    return clf_factory


@Storable.make_storable(StorableType.EXPERIMENT)
class Experiment(Storable):
    def __init__(  # TODO: add tags
        self,
        dataset: 'database.CompleteDataset',
        pool: 'database.Pool',
        cold_start_strategy: strategies.ColdStartStrategy,
        active_learning_strategy: strategies.ActiveLearningStrategy,
    ):
        self.dataset = dataset
        self.pool = pool
        self.cold_start_strategy = cold_start_strategy
        self.active_learning_strategy = active_learning_strategy
        self.histories: dict[int, ExperimentHistory] = {}

    @property
    def budget(self) -> int:
        return self.cold_start_strategy.budget + self.active_learning_strategy.budget  # + 2  # +2 for dirty hack

    @property
    def split(self) -> int:
        return self.cold_start_strategy.budget  # + 2  # +2 for dirty hack

    @property
    def runs(self) -> int:
        return len(self.histories)

    @property
    def sorted_histories(self) -> list[ExperimentHistory]:
        return [h[1] for h in sorted(self.histories.items(), key=lambda p: p[0])[: self.runs]]

    @staticmethod
    def _get_salt(
        dataset_id: 'database.DatasetID',
        pool_id: ID,
        cold_start_strategy: strategies.ColdStartStrategy,
        active_learning_strategy: strategies.ActiveLearningStrategy,
    ) -> Hash:
        return Storable.combine_hashes(
            *map(
                Storable.hash_str,
                itertools.chain(
                    dataset_id,
                    pool_id,
                    map(str, (cold_start_strategy, active_learning_strategy)),
                ),
            )
        )

    def get_id(self) -> ID:
        return self.make_id(
            self.dataset.get_id(),
            self.pool.get_id(),
            self.cold_start_strategy,
            self.active_learning_strategy,
        )

    @staticmethod
    def make_id(
        dataset_id: 'database.DatasetID',
        pool_id: ID,
        cold_start_strategy: strategies.ColdStartStrategy,
        active_learning_strategy: strategies.ActiveLearningStrategy,
    ) -> ID:
        return f'{dataset_id}__{pool_id}__{cold_start_strategy}__{active_learning_strategy}#{Experiment._get_salt(dataset_id, pool_id, cold_start_strategy, active_learning_strategy)}'

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        'dataset': self.dataset.get_id(),
                        'pool': self.pool.get_id(),
                        'cold_start_strategy': str(self.cold_start_strategy),
                        'active_learning_strategy': str(self.active_learning_strategy),
                        'runs': self.runs,
                        'histories': {run: history.get_id() for run, history in self.histories.items()},
                    },
                    type=StorableType.EXPERIMENT,
                    id=self.get_id(),
                ),
                **self.dataset.as_storable().entries,
                **self.pool.as_storable().entries,
                **dict(item for h in self.histories.values() for item in h.as_storable().entries.items()),
            },
        )

    @staticmethod
    def from_storable(
        entry: StorableEntry, data: 'database.DataDatabase', storable_type: 'StorableType | None' = None
    ) -> Self:
        if entry.id in data and storable_type != StorableType.EXPERIMENT:
            return data.retrieve(entry.id)
        payload = entry.payload
        obj = Experiment(
            dataset=data.retrieve(payload['dataset']),
            pool=data.retrieve(payload['pool']),
            cold_start_strategy=strategies.ColdStartStrategy.from_str(payload['cold_start_strategy']),
            active_learning_strategy=strategies.ActiveLearningStrategy.from_str(payload['active_learning_strategy']),
        )
        obj.histories = {int(run): data.retrieve(history_id) for run, history_id in payload['histories'].items()}
        data.encache(obj)
        return obj

    def __repr__(self) -> str:  # TODO: print tags
        return f'Experiment(cs={self.cold_start_strategy.__repr__()}, al={self.active_learning_strategy.__repr__()}, seed={self.pool.indices.seed}, runs={self.runs}, dataset={self.dataset.id})'

    def run_single(
        self,
        tokenizer: 'BertTokenizerFast',
        fine_tuning_model: str,
        run_number: int,
        database: 'database.DataDatabase',
    ):
        from small_text import PoolBasedActiveLearner

        num_classes = len(np.unique(self.dataset.train.y))
        train_dataset = self.pool.to_transformers_dataset(tokenizer, num_classes)
        test_dataset = self.dataset.validation.to_transformers_dataset(tokenizer, num_classes)
        active_learner = PoolBasedActiveLearner(
            make_classifier_factory(
                fine_tuning_model,
                num_classes,
                train_batch_size=10,
            ),
            strategies.ComposeStrategyWrapper(self.active_learning_strategy, self.cold_start_strategy, num_classes),
            train_dataset,
        )

        clf = active_learner._clf_factory.new()
        clf.initialize()
        active_learner.initialize(clf)

        clf = active_learner._clf
        orig_embed = getattr(clf, 'embed', None)
        if callable(orig_embed):

            def embed_adapter(*args, **kwargs):
                kwargs.pop('pbar', None)
                return orig_embed(*args, **kwargs)

            clf.embed = embed_adapter

        indices_labeled = np.array([], dtype=np.int64)

        # cold start cycle
        print('[INFO]: Running cold start strategy...')
        acc_cs, macro_f1_cs, duration_cs, indices_labeled = self.cold_start_strategy.query_strategy.run_loop(
            self.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.cold_start_strategy.n_iterations,
            batch_size=self.cold_start_strategy.batch_size,
        )

        # active learning cycle
        print('[INFO]: Running active learning strategy...')
        acc_al, macro_f1_al, duration_al, indices_labeled = self.active_learning_strategy.query_strategy.run_loop(
            self.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.active_learning_strategy.n_iterations,
            batch_size=self.active_learning_strategy.batch_size,
        )

        self.histories[run_number] = ExperimentHistory(
            final_accuracy=acc_al,
            final_macro_f1=macro_f1_al,
            after_cold_start_accuracy=acc_cs,
            after_cold_start_macro_f1=macro_f1_cs,
            dataset_id=self.dataset.id,
            pool_size=self.pool.size,
            seed=self.pool.indices.seed,
            cs_strategy=self.cold_start_strategy,
            al_strategy=self.active_learning_strategy,
            duration_cs=duration_cs,
            duration_total=duration_cs + duration_al,
        )

        database.store_fast(self.histories[run_number].as_storable(), Format.JSON)

    def run(
        self,
        tokenizer: 'BertTokenizerFast',
        fine_tuning_model: str,
        database: 'database.DataDatabase',
        *,
        repeat: int = 1,
        from_run: int = 1,
        run_iteration_callback: Callable[[Self, int], None] | None = None,
    ):
        for run in range(from_run, repeat + 1):
            if run_iteration_callback is not None:
                run_iteration_callback(self, run)
            self.run_single(
                tokenizer,
                fine_tuning_model,
                run,
                database,
            )


@Storable.make_storable(StorableType.EXPERIMENTS)
class Experiments(Storable):
    @dataclasses.dataclass(frozen=True, eq=True)
    class ExperimentKey:
        dataset_id: 'database.DatasetID'
        pool_size: int
        seed: int
        split: int
        budget: int
        cs_strategy: strategies.ColdStartStrategy
        al_strategy: strategies.QueryStrategyType

        @staticmethod
        def from_experiment(experiment: Experiment) -> 'Experiments.ExperimentKey':
            return Experiments.ExperimentKey(
                dataset_id=experiment.dataset.id,
                pool_size=len(experiment.pool),
                seed=getattr(experiment.pool.indices, 'seed', -1),
                split=experiment.split,
                budget=experiment.budget,
                cs_strategy=experiment.cold_start_strategy,
                al_strategy=experiment.active_learning_strategy.query_strategy,
            )

        @staticmethod
        def empty() -> 'Experiments.ExperimentKey':
            return Experiments.ExperimentKey(None, None, None, None, None, None, None)

        def equivalent(self, other: 'Experiments.ExperimentKey') -> bool:
            return (
                (self.dataset_id is None or other.dataset_id is None or self.dataset_id == other.dataset_id)
                and (self.pool_size is None or other.pool_size is None or self.pool_size == other.pool_size)
                and (self.seed is None or other.seed is None or self.seed == other.seed)
                and (self.split is None or other.split is None or self.split == other.split)
                and (self.budget is None or other.budget is None or self.budget == other.budget)
                and (self.cs_strategy is None or other.cs_strategy is None or self.cs_strategy == other.cs_strategy)
                and (self.al_strategy is None or other.al_strategy is None or self.al_strategy == other.al_strategy)
            )

        def __eq__(self, value):
            if not isinstance(value, Experiments.ExperimentKey):
                return False
            return (
                self.dataset_id == value.dataset_id
                and self.dataset_id is not None
                and self.pool_size == value.pool_size
                and self.pool_size is not None
                and self.seed == value.seed
                and self.seed is not None
                and self.split == value.split
                and self.split is not None
                and self.budget == value.budget
                and self.budget is not None
                and self.cs_strategy == value.cs_strategy
                and self.cs_strategy is not None
                and self.al_strategy == value.al_strategy
                and self.al_strategy is not None
            )

        # def set(self, property, value) -> "Experiments.ExperimentKey":
        #     return dataclasses.replace(self, **{property: value})

    @dataclasses.dataclass(eq=True)
    class ExperimentInfo:
        force: bool | None = None

    def __init__(self, *experiments: Experiment):
        self.experiments = list(experiments)
        self.__experiments_map = {
            Experiments.ExperimentKey.from_experiment(exp): (exp, i, Experiments.ExperimentInfo())
            for i, exp in enumerate(experiments)
        }

    def __iter__(self):
        return iter(self.experiments)

    def __len__(self):
        return len(self.experiments)

    @staticmethod
    def _get_salt() -> Hash:
        return Storable.hash_str('experminets')

    def get_id(self) -> ID:
        raise ValueError('Experminets should not be stored out of database managment system')

    @staticmethod
    def make_id() -> ID:
        return f'experiments#{Experiments._get_salt()}'

    def get_global_id(self) -> ID:
        return self.make_id()

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_global_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        'experiments': [experiment.get_id() for experiment in self.experiments],
                    },
                    type=StorableType.EXPERIMENTS,
                ),
                **dict(item for e in self.experiments for item in e.as_storable().entries.items()),
            },
        )

    @staticmethod
    def from_storable(
        entry: StorableEntry, data: 'database.DataDatabase', storable_type: 'StorableType | None' = None
    ) -> Self:
        if entry.id in data and storable_type != StorableType.EXPERIMENTS:
            return data.retrieve(entry.id)
        return Experiments(*(data.retrieve(experiment_id) for experiment_id in entry.payload['experiments']))

    def merge(self, other: 'Experiments') -> 'Experiments':
        self.__experiments_map.update(other.__experiments_map)
        self.experiments = [exp for exp, _, _ in sorted(self.__experiments_map.values(), key=lambda x: x[1])]
        return self

    def set(self, obj: 'database.DataDatabase | Experiment') -> Self:
        if isinstance(obj, database.DataDatabase):
            self.database = obj
        elif isinstance(obj, Experiment):
            key = Experiments.ExperimentKey.from_experiment(obj)
            if key not in self.__experiments_map:
                self.add(obj)
            else:
                _, index, info = self.__experiments_map[key]
                self.experiments[index] = obj
                self.__experiments_map[key] = (obj, index, info)
        return self

    def add(self, experiment: Experiment):
        key = Experiments.ExperimentKey.from_experiment(experiment)
        if key in self.__experiments_map:
            raise ValueError('Experiment already exists in the collection')
        self.experiments.append(experiment)
        self.__experiments_map[key] = (experiment, len(self.experiments) - 1, Experiments.ExperimentInfo())

    @staticmethod
    def from_experiments(*experiments: 'Experiments') -> 'Experiments':
        return functools.reduce(lambda acc, exp: acc.merge(exp), experiments, Experiments())

    @staticmethod
    @carried_partial_apply
    def from_product(
        database: 'database.DataDatabase',
        datasets: list['database.CompleteDataset'],
        seeds: list[int],
        pool_size: int,
        cold_start_strategies: list[strategies.QueryStrategyType],
        active_learning_strategies: list[strategies.QueryStrategyType],
        bugdets: list[int],
        splits: list[int],
        al_batch_size: int,
    ) -> 'Experiments':
        return Experiments(
            *(
                Experiment(
                    dataset=dataset,
                    pool=dataset.pool(
                        Storable.storable_factory(
                            database, StorableType.SEEDED_INDICES, pool_size, seed, dataset.len_train
                        )
                    ),
                    cold_start_strategy=strategies.ColdStartStrategy(cs_strategy_type, batch_size=split),
                    active_learning_strategy=strategies.ActiveLearningStrategy(
                        al_strategy_type, al_batch_size, budget - split
                    ),
                )
                for dataset, seed, cs_strategy_type, al_strategy_type, budget, split in itertools.product(
                    datasets,
                    seeds,
                    cold_start_strategies,
                    active_learning_strategies,
                    bugdets,
                    splits,
                )
            )
        )

    def run_all(
        self,
        tokenizer: 'BertTokenizerFast',
        fine_tuning_model: str,
        runs: int,
        database: 'database.DataDatabase',
        *,
        default_force: bool = False,
        experiment_iteration_callback: Callable[[int, Experiment, ExperimentInfo], None] | None = None,
        run_iteration_callback: Callable[[Experiment, int], None] | None = None,
    ):
        for i, experiment in enumerate(self.experiments):
            experiment_info = self.__experiments_map[Experiments.ExperimentKey.from_experiment(experiment)][2]
            force = default_force if experiment_info.force is None else experiment_info.force
            exp: Experiment = database.get_experiment(experiment)
            if exp.runs >= runs and not force:
                self.set(exp)
            else:
                starting_point = exp.runs + 1 if exp.runs > 0 else 1
                if experiment_iteration_callback is not None:
                    experiment_iteration_callback(i, exp, experiment_info)
                exp.run(
                    tokenizer,
                    fine_tuning_model,
                    database,
                    repeat=runs,
                    from_run=starting_point,
                    run_iteration_callback=run_iteration_callback,
                )
                database.store_fast(exp.as_storable(), Format.JSON)
                self.set(exp)

    def __getitem__(self, key: 'Experiments.ExperimentKey') -> tuple[Experiment, int, 'Experiments.ExperimentInfo']:
        matching = {e for e in self.__experiments_map.keys() if e.equivalent(key)}
        if len(matching) != 1:
            raise KeyError(f'Key {key} is ambiguous, matches {len(matching)} experiments')
        return self.__experiments_map[matching.pop()]

    def __contains__(self, key: 'Experiments.ExperimentKey | Experiment') -> bool:
        if isinstance(key, Experiment):
            key = Experiments.ExperimentKey.from_experiment(key)
        return any(e == key for e in self.__experiments_map.keys())

    def __delitem__(self, key: 'Experiments.ExperimentKey'):
        matching = {e for e in self.__experiments_map.keys() if e.equivalent(key)}
        if len(matching) != 1:
            raise KeyError(f'Key {key} is ambiguous, matches {len(matching)} experiments')
        key = matching.pop()
        _, idx, _ = self.__experiments_map[key]
        del self.__experiments_map[key]
        tmp = tuple(self.__experiments_map.items())
        for k, (exp, i, info) in tmp:
            if i > idx:
                self.__experiments_map[k] = (exp, i - 1, info)
        self.experiments.pop(idx)

    def matching(
        self, key: 'Experiments.ExperimentKey'
    ) -> builtins.set[tuple[Experiment, int, 'Experiments.ExperimentInfo']]:
        return {self.__experiments_map[e] for e in self.__experiments_map.keys() if e.equivalent(key)}

    def sort_by(self, *, first: Iterable[str | dict] | None = None, last: Iterable[str | dict] | None = None):
        if first is None:
            first = ()
        if last is None:
            last = ()
        VALID_PARAMS = {'dataset', 'seed', 'cold_start_strategy', 'active_learning_strategy', 'budget', 'split'}
        first = tuple(first)
        last = tuple(last)
        first_names = tuple(attr if isinstance(attr, str) else attr['attr'] for attr in first)
        last_names = tuple(attr if isinstance(attr, str) else attr['attr'] for attr in last)
        if set(first_names) & set(last_names):
            raise ValueError(f'Can\'t order both first and last by {set(first) & set(last)}')
        if len(set(first_names)) != len(first) or len(set(last_names)) != len(last):
            raise ValueError('Can\'t order by repeating keys')
        if not all(p in VALID_PARAMS for p in itertools.chain(first_names, last_names)):
            raise ValueError('Unknown sorting criteria')

        def get_param_key(exp: Experiment, param: str, reverse=False):
            return reverse_key(*extract_param(exp, param), reverse)

        def extract_param(exp: Experiment, param: str) -> tuple[int, type[int]] | tuple[str, type[str]]:
            if param == 'dataset':
                return str(exp.dataset.id), str
            elif param == 'seed':
                return exp.pool.indices.seed, int
            elif param == 'cold_start_strategy':
                return str(exp.cold_start_strategy.query_strategy), str
            elif param == 'active_learning_strategy':
                return str(exp.active_learning_strategy.query_strategy), str
            elif param == 'budget':
                return exp.budget, int
            elif param == 'split':
                return exp.split, int
            raise ValueError(f'Unknown parameter: {param}')

        def reverse_key(key: str | int, type: type[str] | type[int], reverse: bool):
            if not reverse:
                return key
            if type is int:
                return -key
            elif type is str:

                @functools.total_ordering
                class ReverseStrKey:
                    def __init__(self, key: str):
                        self.key = key

                    def __eq__(self, value: Self):
                        return self.key == value.key

                    def __lt__(self, value: Self):
                        return self.key > value.key

                return ReverseStrKey(key)
            raise ValueError(f'Unknown type: {type}')

        props = last[::-1] + tuple(VALID_PARAMS - set(first_names) - set(last_names)) + first[::-1]
        key_func = lambda exp: tuple(
            (
                get_param_key(exp, param)
                if isinstance(param, str)
                else get_param_key(exp, param['attr'], reverse=param.get('reverse', False))
            )
            for param in props
        )
        self.experiments.sort(key=key_func)
        for i, exp in enumerate(self.experiments):
            self.__experiments_map[Experiments.ExperimentKey.from_experiment(exp)] = (
                exp,
                i,
                self.__experiments_map[Experiments.ExperimentKey.from_experiment(exp)][2],
            )
