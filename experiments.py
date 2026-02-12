from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

import strategies
import database
from storage import Storable, StorableBundle, StorableEntry, StorableType, ID, Hash, Format

from small_text import (
    TransformerModelArguments,
    TransformerBasedClassificationFactory,
    random_initialization_balanced,
)
import numpy as np
import datetime

import uuid
import builtins
import dataclasses
import itertools
import functools
from typing import Self, Optional, Never


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
        cs_strategy: strategies.QueryStrategyType,
        al_strategy: strategies.QueryStrategyType,
    ) -> Hash:
        return Storable.combine_hashes(
            *map(
                Storable.hash_str,
                map(str, (dataset_id, seed, cs_strategy, al_strategy)),  # TODO: strategy hashing
            )
        )

    def get_id(self) -> ID:
        return f"{self.uuid}#{self._get_salt(self.dataset_id, self.seed, self.cs_strategy, self.al_strategy)}"

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        "final_accuracy": self.final_accuracy,
                        "final_macro_f1": self.final_macro_f1,
                        "after_cold_start_accuracy": self.after_cold_start_accuracy,
                        "after_cold_start_macro_f1": self.after_cold_start_macro_f1,
                        "dataset_id": str(self.dataset_id),
                        "pool_size": self.pool_size,
                        "seed": self.seed,
                        "cs_strategy": str(self.cs_strategy),
                        "al_strategy": str(self.al_strategy),
                        "format_version": self.format_version,
                        "uuid": self.uuid,
                        "duration_cs": self.duration_cs / datetime.timedelta(milliseconds=1),
                        "duration_total": self.duration_total / datetime.timedelta(milliseconds=1),
                    },
                    type=StorableType.EXPERIMENT_HISTORY,
                    id=self.get_id(),
                )
            },
        )

    # @staticmethod
    # def from_json(json_dict: dict[str, any]) -> "ExperimentHistory":
    #     return ExperimentHistory(
    #         final_accuracy=json_dict["final_accuracy"],
    #         final_macro_f1=json_dict["final_macro_f1"],
    #         after_cold_start_accuracy=json_dict["after_cold_start_accuracy"],
    #         after_cold_start_macro_f1=json_dict["after_cold_start_macro_f1"],
    #         dataset_id=database.DatasetID.from_str(json_dict["dataset_id"]),
    #         pool_size=json_dict["pool_size"],
    #         seed=json_dict["seed"],
    #         cs_strategy=strategies.ColdStartStrategy.from_str(json_dict["cs_strategy"]),
    #         al_strategy=strategies.ActiveLearningStrategy.from_str(json_dict["al_strategy"]),
    #         format_version=json_dict.get("format_version", 1),
    #         duration_cs=datetime.timedelta(milliseconds=json_dict["duration_cs"]),
    #         duration_total=datetime.timedelta(milliseconds=json_dict["duration_total"]),
    #     )


def make_classifier_factory(
    transformer_model_name: str, num_classes: int, train_batch_size: int, num_epochs: int = 3
) -> TransformerBasedClassificationFactory:
    model_args = TransformerModelArguments(transformer_model_name)

    import torch
    
    clf_factory = TransformerBasedClassificationFactory(
        model_args,
        num_classes,
        kwargs=dict(
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_epochs=num_epochs,
            mini_batch_size=train_batch_size,
        ),
    )
    return clf_factory


class Experiment(Storable):  # TODO: multiple runs
    def __init__(  # TODO: add tags
        self,
        dataset: "database.CompleteDataset",
        pool: "database.Pool",
        cold_start_strategy: strategies.ColdStartStrategy,
        active_learning_strategy: strategies.ActiveLearningStrategy,
        # name: str | None = None,
    ):
        self.dataset = dataset
        self.pool = pool
        self.cold_start_strategy = cold_start_strategy
        self.active_learning_strategy = active_learning_strategy
        # self.name = (
        #     name
        #     if name is not None
        #     else f"{dataset.get_id()}_cs={cold_start_strategy}_al={active_learning_strategy}_{pool}"
        # )
        self.histories:  dict[int, ExperimentHistory] = {}

    @property
    def budget(self) -> int:
        return self.cold_start_strategy.budget + self.active_learning_strategy.budget + 2  # +2 for dirty hack

    @property
    def split(self) -> int:
        return self.cold_start_strategy.budget + 2  # +2 for dirty hack

    @property
    def runs(self) -> int:
        return len(self.histories)

    @staticmethod
    def _get_salt(dataset, pool, cold_start_strategy, active_learning_strategy, runs) -> Hash:
        return Storable.combine_hashes(
            *map(
                Storable.hash_str,
                itertools.chain(
                    dataset.get_id(),
                    pool.get_id(),
                    map(str, (cold_start_strategy, active_learning_strategy, runs)),
                ),
            )
        )

    def get_id(self) -> ID:
        return f"{self.dataset.get_id()}__{self.pool.get_id()}__{self.cold_start_strategy}__{self.active_learning_strategy}__{self.runs}#{self._get_salt(self.dataset, self.pool, self.cold_start_strategy, self.active_learning_strategy, self.runs)}"

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        "dataset": self.dataset.get_id(),
                        "pool": self.pool.get_id(),
                        "cold_start_strategy": str(self.cold_start_strategy),
                        "active_learning_strategy": str(self.active_learning_strategy),
                        "runs": self.runs,
                        "histories": {run: history.get_id() for run, history in self.histories.items()},
                    },
                    type=StorableType.EXPERIMENT,
                    id=self.get_id(),
                ),
                **self.dataset.as_storable().entries,
                **self.pool.as_storable().entries,
                **dict(item for h in self.histories.values() for item in h.as_storable().entries.items()),
            },
        )

    # def dump(self, root_dir: None = None, filename: str | None = None):
    #     if self.__history is None:
    #         raise ValueError("Experiment was not run yet, cannot dump")
    #     if root_dir is not None:
    #         raise ValueError("Root dir cannot be changed when dumping Experiment")

    #     config = self.get_config()

    #     with (
    #         open_subbuild(
    #             self.root_dir, database.DATA_DIR, filename if filename is not None else self.get_config_filename()
    #         ).open("w") as ef,
    #         open_subbuild(self.root_dir, config["history"]).open("w") as hf,
    #     ):
    #         json.dump(config, ef)
    #         json.dump(self.history.to_json(), hf)

    #     self.dataset.dump(self.root_dir)
    #     self.pool.dump(self.root_dir, f"{self.dataset.get_id()}_{self.pool.indices.repr['seed']}_{self.pool.size}")

    # @staticmethod
    # def load(root_dir: pathlib.Path, filename: str) -> "Experiment":

    #     with pathlib.Path(root_dir, database.DATA_DIR, filename).open("r") as ef:
    #         config = json.load(ef)
    #     dataset = database.CompleteDataset.from_json(config["dataset"])
    #     pool = database.Pool.load(root_dir, config["pool"], dataset)
    #     cold_start_strategy = strategies.ColdStartStrategy.from_str(config["cold_start_strategy"])
    #     active_learning_strategy = strategies.ActiveLearningStrategy.from_str(config["active_learning_strategy"])
    #     experiment = Experiment(
    #         dataset=dataset,
    #         pool=pool,
    #         cold_start_strategy=cold_start_strategy,
    #         active_learning_strategy=active_learning_strategy,
    #         root_dir=root_dir,
    #         name=config["name"],
    #     )
    #     with pathlib.Path(root_dir, config["history"]).open("r") as hf:  # TODO : подумать над другим форматом
    #         history_json = json.load(hf)
    #     experiment.__history = ExperimentHistory.from_json(history_json)
    #     return experiment

    # def __exit__(self, exc_type, exc_value, traceback):
    #     if self.__history is None:
    #         print("Warning: Experiment was not run, no history to save.")
    #     else:
    #         self.dump()
    #         if self.__database is not None:
    #             self.__database.register(self)

    def __repr__(self) -> str:  # TODO: print tags
        return f"Experiment(cs={self.cold_start_strategy.__repr__()}, al={self.active_learning_strategy.__repr__()}, seed={self.pool.indices.seed}, runs={self.runs}, dataset={self.dataset.id})"

    def run_single(self, tokenizer: "BertTokenizerFast", fine_tuning_model: str, run_number: int, database: "database.DataDatabase"):
        from small_text import PoolBasedActiveLearner
        print(f"=== RUNNING EXPERIMENT (run {run_number}): {self} ===")
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
        # TODO: FIX DIRTY HACK

        indices_initial = random_initialization_balanced(train_dataset.y, n_samples=2)
        y_initial = np.array(train_dataset.y[indices_initial], dtype=np.int64)
        active_learner.initialize_data(indices_initial, y_initial)

        indices_labeled = indices_initial.copy()

        # cold start cycle
        print("Running cold start strategy...")
        acc_cs, macro_f1_cs, duration_cs, indices_labeled = self.cold_start_strategy.query_strategy.run_loop(
            self.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.cold_start_strategy.n_iterations,
            batch_size=self.cold_start_strategy.batch_size,
        )
        # active learning cycle
        print("Running active learning strategy...")
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

    def run(self, tokenizer: "BertTokenizerFast", fine_tuning_model: str, database: "database.DataDatabase", *, repeat: int = 1, from_run: int = 1):
        for run in range(from_run, from_run + repeat):
            self.run_single(tokenizer, fine_tuning_model, run, database)


class Experiments(Storable):
    @dataclasses.dataclass(frozen=True, eq=True)
    class ExperimentKey:
        dataset_id: "database.DatasetID"
        pool_size: int
        seed: int
        split: int
        budget: int
        cs_strategy: strategies.ColdStartStrategy
        al_strategy: strategies.QueryStrategyType

        @staticmethod
        def from_experiment(experiment: Experiment) -> "Experiments.ExperimentKey":
            return Experiments.ExperimentKey(
                dataset_id=experiment.dataset.id,
                pool_size=experiment.pool.size,
                seed=getattr(experiment.pool.indices, "seed", -1),
                split=experiment.split,
                budget=experiment.budget,
                cs_strategy=experiment.cold_start_strategy,
                al_strategy=experiment.active_learning_strategy.query_strategy,
            )

        @staticmethod
        def empty() -> "Experiments.ExperimentKey":
            return Experiments.ExperimentKey(None, None, None, None, None, None, None)

        def equivalent(self, other: "Experiments.ExperimentKey") -> bool:
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
        return Storable.hash_str("experminets")

    def get_id(self) -> ID:
        raise ValueError('Experminets should not be stored out of database managment system')

    def get_global_id(self) -> ID:
        return f"experiments#{self._get_salt()}"

    def as_storable(self) -> StorableBundle:
        return StorableBundle(
            main=self.get_global_id(),
            entries={
                self.get_id(): StorableEntry(
                    payload={
                        "expeiments": [experiment.get_id() for experiment in self.experiments],
                    },
                    type=StorableType.EXPERIMENTS,
                ),
                **dict(item for e in self.experiments for item in e.as_storable().entries.items()),
            },
        )

    def merge(self, other: "Experiments") -> "Experiments":
        self.__experiments_map.update(other.__experiments_map)
        self.experiments = [exp for exp, _, _ in sorted(self.__experiments_map.values(), key=lambda x: x[1])]
        return self

    def set(self, obj: "database.DataDatabase | Experiment") -> Self:
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
            raise ValueError("Experiment already exists in the collection")
        self.experiments.append(experiment)
        self.__experiments_map[key] = (experiment, len(self.experiments) - 1, Experiments.ExperimentInfo())

    @staticmethod
    def from_experiments(*experiments: "Experiments") -> "Experiments":
        return functools.reduce(lambda acc, exp: acc.merge(exp), experiments, Experiments())

    @staticmethod
    def from_product(
        datasets: list['database.CompleteDataset'],
        seeds: list[int],
        pool_size: int,
        cold_start_strategies: list[strategies.QueryStrategyType],
        active_learning_strategies: list[strategies.QueryStrategyType],
        bugdets: list[int],
        splits: list[int],
        al_batch_size: int,
    ) -> "Experiments":
        return Experiments(
            *(
                (dataset.create_pool(database.SeededIndices(pool_size, seed, len(dataset.train))) and None) or Experiment(
                    dataset=dataset,
                    pool=dataset.pool,
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

    def run_all(self, tokenizer: "BertTokenizerFast", fine_tuning_model: str, runs: int, database: "database.DataDatabase", *, default_force: bool = False):
        for experiment in self.experiments:
            experiment_info = self.__experiments_map[Experiments.ExperimentKey.from_experiment(experiment)][2]
            force = default_force if experiment_info.force is None else experiment_info.force
            exp: Experiment = database.retrieve(experiment)
            if exp.runs >= runs and not force:
                self.set(exp)
            else:
                starting_point = exp.runs + 1 if exp.runs > 0 else 1
                exp.run(tokenizer, fine_tuning_model, database, repeat=runs, from_run=starting_point)
                database.store_fast(exp.as_storable(), Format.JSON)
                database.experiments.add(exp) # TODO: move to from_bundle

    def __getitem__(self, key: "Experiments.ExperimentKey") -> tuple[Experiment, int, "Experiments.ExperimentInfo"]:
        matching = {e for e in self.__experiments_map.keys() if e.equivalent(key)}
        if len(matching) != 1:
            raise KeyError(f"Key {key} is ambiguous, matches {len(matching)} experiments")
        return self.__experiments_map[matching.pop()]

    def __contains__(self, key: "Experiments.ExperimentKey | Experiment") -> bool:
        if isinstance(key, Experiment):
            key = Experiments.ExperimentKey.from_experiment(key)
        return any(e == key for e in self.__experiments_map.keys())

    def matching(
        self, key: "Experiments.ExperimentKey"
    ) -> builtins.set[tuple[Experiment, int, "Experiments.ExperimentInfo"]]:
        return {self.__experiments_map[e] for e in self.__experiments_map.keys() if e.equivalent(key)}

    # def to_json(self) -> dict:
    #     return {
    #         "experiments": {
    #             i: {
    #                 "experiment": str(database.DATA_DIR / experiment.get_config_filename()),
    #                 "info": info.to_json(),
    #                 "key": key.to_json(),
    #             }
    #             for key, (experiment, i, info) in self.__experiments_map.items()
    #         },
    #         "root_dir": self.experiments[0].root_dir,
    #     }

    # def from_json(data: dict) -> "Experiments":
    #     root_dir = data["root_dir"]
    #     return Experiments(
    #         *(
    #             Experiment.load(root_dir, exp_data["experiment"])
    #             for _, exp_data in sorted(data["experiments"].items(), key=lambda x: int(x[0]))
    #         )
    #     )
