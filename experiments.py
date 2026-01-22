import strategies
import database
from utils import Dumpable, JSONifiable

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text import PoolBasedActiveLearner, TransformerModelArguments, TransformerBasedClassificationFactory, random_initialization_balanced
import numpy as np
import datetime
import torch

import dataclasses
import os
import json

@dataclasses.dataclass
class ExperimentHistory(JSONifiable):
    final_accuracy: float
    final_macro_f1: float
    after_cold_start_accuracy: float
    after_cold_start_macro_f1: float
    dataset_id: database.DatasetID
    pool_size: int
    seed: int
    cs_strategy: strategies.ColdStartStrategy
    al_strategy: strategies.ActiveLearningStrategy
    duration_cs: datetime.timedelta
    duration_total: datetime.timedelta
    format_version: int = 1

    @property
    def split(self) -> int:
        return self.cs_strategy.budget
    
    @property
    def budget(self) -> int:
        return self.cs_strategy.budget + self.al_strategy.budget
    
    def to_json(self) -> dict[str, any]:
        return {
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
            'duration_cs': self.duration_cs/datetime.timedelta(milliseconds=1),
            'duration_total': self.duration_total/datetime.timedelta(milliseconds=1),
        }
    
    @staticmethod
    def from_json(json_dict: dict[str, any]) -> 'ExperimentHistory':
        return ExperimentHistory(
            final_accuracy=json_dict['final_accuracy'],
            final_macro_f1=json_dict['final_macro_f1'],
            after_cold_start_accuracy=json_dict['after_cold_start_accuracy'],
            after_cold_start_macro_f1=json_dict['after_cold_start_macro_f1'],
            dataset_id=database.DatasetID.from_str(json_dict['dataset_id']),
            pool_size=json_dict['pool_size'],
            seed=json_dict['seed'],
            cs_strategy=strategies.ColdStartStrategy.from_str(json_dict['cs_strategy']),
            al_strategy=strategies.ActiveLearningStrategy.from_str(json_dict['al_strategy']),
            format_version=json_dict.get('format_version', 1),
            duration_cs=datetime.timedelta(milliseconds=json_dict['duration_cs']),
            duration_total=datetime.timedelta(milliseconds=json_dict['duration_total']),
        )
    
def make_classifier_factory(transformer_model_name: str, num_classes: int, train_batch_size: int, num_epochs: int = 3) -> TransformerBasedClassificationFactory:
    model_args = TransformerModelArguments(transformer_model_name)
    
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

class Experiment(Dumpable):
    def __init__(
        self,
        dataset: database.Dataset,
        pool: database.Pool,
        cold_start_strategy: strategies.ColdStartStrategy,
        active_learning_strategy: strategies.ActiveLearningStrategy,
        root_dir: os.PathLike,
        name: str | None = None,
    ):
        self.dataset = dataset
        self.pool = pool
        self.cold_start_strategy = cold_start_strategy
        self.active_learning_strategy = active_learning_strategy
        self.root_dir = root_dir
        self.name = name if name is not None else f"{dataset.get_id()}_cs={cold_start_strategy}_al={active_learning_strategy}_{pool}"
        self.__history = None

    def get_config_filename(self, external_id: str | None = None) -> str:
        if external_id is not None:
            raise ValueError("External ID is not supported for Experiment config filename")
        return f"{self.name}_experiment.json" 
    
    def get_config(self, external_id: str | None = None):
        if external_id is not None:
            raise ValueError("External ID is not supported for Experiment config")
        if self.__history is None:
            raise ValueError("Experiment was not run yet, cannot get config")
        return {
            'dataset': self.dataset.to_json(),
            'pool': os.path.join(database.DATA_DIR, self.pool.get_config_filename(self.name)),
            'cold_start_strategy': str(self.cold_start_strategy),
            'active_learning_strategy': str(self.active_learning_strategy),
            'name': self.name,
            'history': os.path.join(database.DATA_DIR, f"{self.name}_history.json")
        }
        
    def dump(self, root_dir: os.PathLike | None = None, filename: str | None = None):
        if self.__history is None:
            raise ValueError("Experiment was not run yet, cannot dump")
        if root_dir is not None:
            raise ValueError("Root dir cannot be changed when dumping Experiment")
        
        config = self.get_config()
        with (open(os.path.join(self.root_dir, 
                                database.DATA_DIR, 
                                filename if filename is not None else self.get_config_filename()
                                ), 'w') as ef, 
              open(os.path.join(self.root_dir, 
                                config['history']
                                ), 'w') as hf
             ):
            json.dump(config, ef)
            json.dump(self.history.to_json(), hf)
        
        self.dataset.dump(self.root_dir)
        self.pool.dump(self.root_dir, self.name)

    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> 'Experiment':
        with open(os.path.join(root_dir, database.DATA_DIR, filename), 'r') as ef:
            config = json.load(ef)
        dataset =  database.Dataset.from_json(config['dataset'])
        pool = database.Pool.load(root_dir, config['pool'], dataset)
        cold_start_strategy =  strategies.ColdStartStrategy.from_str(config['cold_start_strategy'])
        active_learning_strategy = strategies.ActiveLearningStrategy.from_str(config['active_learning_strategy'])
        experiment = Experiment(
            dataset=dataset,
            pool=pool,
            cold_start_strategy=cold_start_strategy,
            active_learning_strategy=active_learning_strategy,
            root_dir=root_dir,
            name=config['name']
        )
        with open(os.path.join(root_dir, config['history']), 'r') as hf: # TODO : подумать над другим форматом
            history_json = json.load(hf)
        experiment.__history = ExperimentHistory.from_json(history_json)
        return experiment

    def __enter__(self) -> 'Experiment':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.__history is None:
            print("Warning: Experiment was not run, no history to save.")
        else:
            self.dump()

    @property
    def budget(self) -> int:
        return self.cold_start_strategy.budget + self.active_learning_strategy.budget + 1 # +1 for dirty hack

    @property 
    def split(self) -> int:
        return self.cold_start_strategy.budget + 1 # +1 for dirty hack
    
    @property
    def history(self) -> ExperimentHistory:
        if self.__history is None:
            raise ValueError("Experiment history is not available. Run the experiment first.")
        return self.__history
    
    def run(self, tokenizer: BertTokenizerFast, fine_tuning_model: str, *, force: bool = False):
        if os.path.exists(os.path.join(self.root_dir, database.DATA_DIR, self.get_config_filename())) and not force:
            with open(os.path.join(self.root_dir, database.DATA_DIR, self.get_config_filename()), 'r') as ef:
                config = json.load(ef)
            if self.__history is None and os.path.exists(os.path.join(self.root_dir, config['history'])):
                with open(os.path.join(self.root_dir, config['history']), 'r') as hf:
                    history_json = json.load(hf)
                self.__history = ExperimentHistory.from_json(history_json)
                return
    
        print(f"=== RUNNING EXPERIMENT: {self.name} ===")
        num_classes = len(set(self.dataset.dataset.train[self.dataset.label_field]))
        train_dataset = database.DatasetView.to_transformers_dataset(
            self.pool.pool.train, 
            tokenizer, 
            num_classes, 
        )
        test_dataset = database.DatasetView.to_transformers_dataset(
            self.pool.pool.validation, 
            tokenizer, 
            num_classes, 
        )
        active_learner = PoolBasedActiveLearner(
            make_classifier_factory(
                fine_tuning_model, 
                num_classes,
                train_batch_size=10,
            ),
            strategies.ComposeStrategyWrapper(
                self.active_learning_strategy,
                self.cold_start_strategy
             ), 
            train_dataset
        )
        # TODO: FIX DIRTY HACK

        indices_initial = random_initialization_balanced(
            self.pool.pool.train.y, n_samples=1
        )
        y_initial = np.array([self.pool.pool.train.y[indices_initial[0]]], dtype=np.int64)
        active_learner.initialize_data(indices_initial, y_initial)


        indices_labeled = indices_initial.copy()

        # cold start cycle
        print("Running cold start strategy...")
        acc_cs, macro_f1_cs, duration_cs, indices_labeled = self.cold_start_strategy.query_strategy.run_loop(
            self.pool.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.cold_start_strategy.n_iterations,
            batch_size=self.cold_start_strategy.batch_size,
        )
        # active learning cycle
        print("Running active learning strategy...")
        acc_al, macro_f1_al, duration_al, indices_labeled = self.active_learning_strategy.query_strategy.run_loop(
            self.pool.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.active_learning_strategy.n_iterations,
            batch_size=self.active_learning_strategy.batch_size,
        )

        assert self.pool.indices.repr is not None 
        
        self.__history = ExperimentHistory(
            final_accuracy=acc_al,
            final_macro_f1=macro_f1_al,
            after_cold_start_accuracy=acc_cs,
            after_cold_start_macro_f1=macro_f1_cs,
            dataset_id=self.dataset.id,
            pool_size=self.pool.size,
            seed=self.pool.indices.repr['seed'],
            cs_strategy=self.cold_start_strategy,
            al_strategy=self.active_learning_strategy,
            duration_cs=duration_cs,
            duration_total=duration_cs + duration_al,
        ) 