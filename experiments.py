import strategies
import database

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from small_text import PoolBasedActiveLearner, TransformerModelArguments, TransformerBasedClassificationFactory, random_initialization_balanced
from small_text.integrations.transformers.datasets import TransformersDataset
import numpy as np
import datetime
import torch

import dataclasses
import os
import json
import time


def run_al_experiment_one(
    tokenizer, 
    dataset_key: str,
    standardized_datasets: Dict[str, Dataset],
    strategy_name: QueryStrategy,
    cfg: ALConfig,
    experiment_name: str | None = None,
    save_checkpoints: bool = False,
    save_history_flag: bool = True,
    oracle_strategy: OracleStrategy | None = None,      # 'human' | 'llm' | 'hybrid'
):
    """
    dataset_key: 'sst2' / 'ag_news'
    strategy_name: 'random', 'least_conf', 'bald', 'badge'
    """
    if experiment_name is None:
        experiment_name = mangle_experiment_name(dataset_key, strategy_name, cfg.seed, oracle_strategy)

    if os.path.exists(os.path.join(c.LOG_DIR, f"{experiment_name}_history.csv")) and cfg.force_cache:
        print(f'Experiment already saved: {experiment_name}')
        return load_history_dict(experiment_name)
    
    print(f"\n=== AL EXPERIMENT: dataset={dataset_key}, strategy={strategy_name}, seed={cfg.seed}, oracle strategy={oracle_strategy} ===")

    ds_std = standardized_datasets[dataset_key]
    train_hf = ds_std["train"]
    eval_hf = get_eval_split(ds_std)

    num_classes = len(set(train_hf["label"]))

    pool_hf, _ = subsample_pool(train_hf, cfg.pool_size, cfg.seed)

    pool_ds = to_transformers_dataset(tokenizer, pool_hf, num_classes, max_length=cfg.max_length)
    test_ds = to_transformers_dataset(tokenizer, eval_hf, num_classes, max_length=cfg.max_length)

    clf_factory = make_classifier_factory(num_classes, cfg)
    query_strategy = make_query_strategy(strategy_name, num_classes)

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, pool_ds)


    # initialization
    indices_initial = random_initialization_balanced(
        pool_ds.y, n_samples=cfg.initial_labeled
    )
    y_initial = np.array([pool_ds.y[i] for i in indices_initial])
    active_learner.initialize_data(indices_initial, y_initial)

    history: List[Dict[str, Any]] = []

    # AL-iterations
    indices_labeled = indices_initial.copy()

    for it in range(cfg.iterations + 1):
        metrics = evaluate_on_test(active_learner, test_ds)
        labeled_count = len(indices_labeled)

        #TODO:  calculate cost properly
        #  oracle = human
        total_cost_human = labeled_count * cfg.cost_human
        total_cost_llm = labeled_count * cfg.cost_llm



        step_info = dict(
            iter=it,
            labeled=labeled_count,
            acc=metrics["acc"],
            macro_f1=metrics["macro_f1"],
            cost_human=total_cost_human,
            cost_llm=total_cost_llm,
        )
        history.append(step_info)

        print(
            f"[Iter {it:02d}] labeled={labeled_count:4d} | "
            f"acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
            f"cost_human={total_cost_human:.1f}"
        )

        if save_checkpoints:
            save_active_learner_checkpoint(tokenizer, active_learner, experiment_name, iteration=it)

        if it == cfg.iterations:
            break

        queried_indices = active_learner.query(num_samples=cfg.batch_size)
        
        if oracle_strategy is not None:
            y_queried = get_oracle_labels(
                queried_indices,
                pool_ds=pool_ds,
                pool_hf=pool_hf,
                dataset_key=dataset_key,
                current_k=it,
                oracle_strategy=oracle_strategy,
            )
        else:  
            y_queried = pool_ds.y[queried_indices]

        active_learner.update(y_queried)
        indices_labeled = np.concatenate([indices_labeled, queried_indices])
    
    if save_history_flag:
      save_history(history, experiment_name)

    return history

@dataclasses.dataclass
class ExperimentHistory:
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

class Experiment:
    # seed: int
    # dataset: database.Dataset
    # pool: database.Pool
    # split: int 
    # cold_start_strategy: ColdStartStrategy
    # active_learning_strategy: ActiveLearningStrategy
    # macro_f1: float
    # accuracy: float

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

    def dump(self):
        if self.__history is None:
            raise ValueError("Experiment was not run yet, cannot dump")
        config = {
            'dataset': { 'id': str(self.dataset.id), 'text_field': self.dataset.text_field, 'label_field': self.dataset.label_field},
            'pool': f"{self.name}_pool.json",
            'cold_start_strategy': str(self.cold_start_strategy),
            'active_learning_strategy': str(self.active_learning_strategy),
            'name': self.name,
            'history': f"{self.name}_history.json"
        }
        with open(os.path.join(self.root_dir, database.DATA_DIR, f"{self.name}_experiment.json"), 'w') as ef, open(os.path.join(self.root_dir, database.DATA_DIR, f"{self.name}_history.json"), 'w') as hf:
            json.dump(config, ef)
            json.dump(self.history.to_json(), hf)
        
        self.dataset.dump(self.root_dir)
        self.pool.dump(self.root_dir, file_name=self.name)

    @staticmethod
    def load(root_dir: os.PathLike, filename: str) -> 'Experiment':
        with open(os.path.join(root_dir, database.DATA_DIR, filename), 'r') as ef:
            config = json.load(ef)
        dataset = database.Dataset(
            id=database.DatasetID.from_str(config['dataset']['id']),
            text_field=config['dataset']['text_field'],
            label_field=config['dataset']['label_field'],
            root_dir=root_dir
        )
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
        with open(os.path.join(root_dir, database.DATA_DIR, config['history']), 'r') as hf:
            history_json = json.load(hf)
        experiment.__history = ExperimentHistory.from_json(history_json)
        return experiment

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
        if os.path.exists(os.path.join(self.root_dir, f"{self.name}_experiment.json")) and not force:
            if self.__history is None and os.path.exists(os.path.join(self.root_dir, database.DATA_DIR, f"{self.name}_history.json")):
                with open(os.path.join(self.root_dir, database.DATA_DIR, f"{self.name}_history.json"), 'r') as hf:
                    history_json = json.load(hf)
                self.__history = ExperimentHistory.from_json(history_json)
                return
    
        print(f"=== RUNNING EXPERIMENT: {self.name} ===")
        num_classes = len(set(self.dataset.dataset.train[self.dataset.label_field]))
        train_dataset = database.to_transformers_dataset(
            tokenizer, 
            self.pool.pool['train'], 
            num_classes, 
        )
        test_dataset = database.to_transformers_dataset(
            tokenizer, 
            self.pool.pool['validation'], 
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
            self.pool.pool.y, n_samples=1
        )
        y_initial = np.array([self.pool.pool.y[i] for i in indices_initial])
        active_learner.initialize_data(indices_initial, y_initial)


        indices_labeled = indices_initial.copy()

        # cold start cycle
        print("Running cold start strategy...")
        acc_cs, macro_f1_cs, duration_cs, indices_labeled = self.cold_start_strategy.query_strategy.run_loop(
            self.pool.pool,
            active_learner,
            indices_labeled,
            test_dataset,
            n_iterations=self.cold_start_strategy.budget // self.active_learning_strategy.batch_size,
            batch_size=self.active_learning_strategy.batch_size,
        )
        # active learning cycle
        print("Running active learning strategy...")
        acc_al, macro_f1_al, duration_al = self.__run_loop(
            tokenizer,
            active_learner,
            self.active_learning_strategy,
            indices_labeled,
            test_dataset
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