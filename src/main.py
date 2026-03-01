from database import *
from experiments import *
from strategies import *
from aggregation import *

import matplotlib.pyplot as plt
import numpy as np
import warnings
from small_text.utils.annotations import ExperimentalWarning
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

import pathlib
from types import MethodType


def main() -> int | None:
    warnings.filterwarnings('ignore', category=ExperimentalWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    disable_progress_bar()

    with DataDatabase(pathlib.Path.cwd().parent) as db:
        db.try_restore()
        ag_news = db.get_dataset(DatasetID('ag_news'))
        sst_2 = db.get_dataset(DatasetID('glue', 'sst2'), text_field='sentence')

        from_product = Experiments.from_product(
            database=db,
            datasets=[ag_news, sst_2],
            # seeds=[42], #, 69, 1337, 1147, 1984],
            seeds=[42, 69, 1337, 1147, 1984],
            pool_size=1000,
            cold_start_strategies=[
                QueryStrategySimple(SimpleQueryStrategyType.RANDOM),
                QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE),
                QueryStrategySimple(SimpleQueryStrategyType.BALD),
                QueryStrategySimple(SimpleQueryStrategyType.BADGE),
            ],
            al_batch_size=10,
        )

        experiments = Experiments.from_experiments(
            from_product(
                active_learning_strategies=[
                    MockQueryStrategyType(),
                ],
                bugdets=[500],
                splits=list(range(10, 300 + 1, 10)),
            ),
            from_product(
                active_learning_strategies=[MockQueryStrategyType()],
                bugdets=[400],
                splits=list(range(10, 300 + 1, 10)),
            ),
        )

        experiments.sort_by(first=('seed', 'dataset'), last=({'attr': 'split', 'reverse': False},))

        tokenizer = AutoTokenizer.from_pretrained(
            'google/bert_uncased_L-2_H-128_A-2', cache_dir=pathlib.Path('./cache')
        )

        def _encode_plus_adapter(self, text, text_pair=None, **kwargs):
            if text_pair is not None:
                return self(text, text_pair, **kwargs)
            return self(text, **kwargs)

        tokenizer.encode_plus = MethodType(_encode_plus_adapter, tokenizer)

        runs = 5

        @dataclasses.dataclass
        class RunningInfo:
            total: int
            at: int = 0
            order_at: int = 0

        info = RunningInfo(sum(exp not in db or db.retrieve(exp.get_id()).runs < runs for exp in experiments))

        def update_info(number: int, exp: Experiment, exp_info: Experiments.ExperimentInfo):
            info.at += 1
            info.order_at = number

        def log_run(exp: Experiment, run: int):
            print(
                f'[INFO]: RUNNING EXPERIMENT [{info.at}#{info.order_at}({run})/{info.total}/{len(experiments)}]: {exp}'
            )

        experiments.run_all(
            tokenizer,
            'google/bert_uncased_L-2_H-128_A-2',
            runs,
            db,
            experiment_iteration_callback=update_info,
            run_iteration_callback=log_run,
        )

        # db.recollect_stored()

        groups = ExperimentGroup.compose_groups(
            ExperimentsRange(
                db.experiments,
                ExperimentFilter(
                    (
                        ExperimentKeyExpression(
                            dataclasses.replace(Experiments.ExperimentKey.empty(), dataset_id=DatasetID('ag_news'))
                        )
                        | ExperimentKeyExpression(
                            dataclasses.replace(Experiments.ExperimentKey.empty(), dataset_id=DatasetID('glue', 'sst2'))
                        )
                    )
                    & ExperimentKeyExpression(
                        dataclasses.replace(Experiments.ExperimentKey.empty(), al_strategy=MockQueryStrategyType())
                    )
                    & ExperimentInExpression(
                        dataclasses.replace(Experiments.ExperimentKey.empty(), split=split)  # + 2)
                        # for split in list(range(10, 300 + 1, 10))
                        for split in list(range(10, 50 + 1, 10))
                    )
                ),
            ),
            [GroupEqFilter('cold_start_strategy.query_strategy'), GroupEqFilter('dataset.id'), GroupEqFilter('split')],
            [[], [], [], []],  # TODO: think about redundant aggregators
            ExperimentSelector(
                [AverageAggregator(), VarianceAggregator()],
                ['final_accuracy', 'final_macro_f1'],
                ['final_accuracy', 'final_macro_f1'],
                runs=5,
            ),
            # ExperimentSelector(
            #     [BundleAggregator()],
            #     ["final_accuracy", "final_macro_f1"],
            #     runs=5,
            # ),
        )

        # print(groups)
        # print(groups.unique_tuples(("seed", "split")))
        printable = groups.to_printable()
        # print(printable)
        # aggregated = groups.aggregate()
        # print(groups.contained[0].contained[0].contained[0])

        strategies = set()
        datasets = set()
        splits = set()
        data = {}

        for strat_key, ds_dict in printable.items():
            strategy_obj = strat_key[0][1]
            strategy_name = str(strategy_obj)
            strategies.add(strategy_name)

            for ds_key, split_dict in ds_dict.items():
                dataset_id = ds_key[0][1]
                if dataset_id.subset:
                    ds_name = f'{dataset_id.path}_{dataset_id.subset}'
                else:
                    ds_name = dataset_id.path
                datasets.add(ds_name)

                for split_key, values in split_dict.items():
                    split = split_key[0][1]
                    splits.add(split)
                    # acc_per_run, f1_per_run = values[0][0]
                    # data[(strategy_name, ds_name, split)] = (acc_per_run, f1_per_run)
                    mean_acc, mean_f1 = values[0][0]
                    var_acc, var_f1 = values[0][1]
                    data[(strategy_name, ds_name, split)] = (mean_acc, mean_f1, var_acc, var_f1)

        datasets = sorted(datasets)
        strategies = sorted(strategies)
        splits_sorted = sorted(splits)

        # runs_count = 5
        # for run_idx in range(runs_count):
        #     fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

        #     for row, metric_name, metric_idx in [(0, "Final Accuracy", 0), (1, "Final Macro F1", 1)]:
        #         for col, ds in enumerate(datasets):
        #             ax = axes[row, col]
        #             for strategy in strategies:
        #                 y = []
        #                 for split in splits_sorted:
        #                     entry = data.get((strategy, ds, split))
        #                     if entry is not None:
        #                         acc_per_run, f1_per_run = entry
        #                         if run_idx < len(acc_per_run):
        #                             value = acc_per_run[run_idx] if metric_idx == 0 else f1_per_run[run_idx]
        #                             y.append(value)
        #                         else:
        #                             y.append(np.nan)
        #                     else:
        #                         y.append(np.nan)
        #                 ax.plot(splits_sorted, y, marker='o', label=strategy)

        #             ax.set_title(f"{ds} – {metric_name} (Run {run_idx+1})")
        #             ax.set_xlabel("Split")
        #             ax.set_ylabel(metric_name)
        #             ax.legend()
        #             ax.grid(True)

        #     plt.tight_layout()
        #     plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

        for row, metric_name, metric_idx, var_idx in [
            (0, 'Final Accuracy', 0, 2),  # mean at index 0, variance at index 2
            (1, 'Final Macro F1', 1, 3),  # mean at index 1, variance at index 3
        ]:
            for col, ds in enumerate(datasets):
                ax = axes[row, col]
                for strategy in strategies:
                    y_mean = []
                    y_std = []
                    for split in splits_sorted:
                        vals = data.get((strategy, ds, split))
                        if vals is not None:
                            y_mean.append(vals[metric_idx])
                            y_std.append(np.sqrt(vals[var_idx]))
                        else:
                            y_mean.append(np.nan)
                            y_std.append(np.nan)
                    ax.errorbar(splits_sorted, y_mean, yerr=y_std, marker='o', label=strategy, capsize=3)
                ax.set_title(f'{ds} – {metric_name}')
                ax.set_xlabel('Split')
                ax.set_ylabel(metric_name)
                ax.legend()
                ax.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    raise SystemExit(main())
