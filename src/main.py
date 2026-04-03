from database import *
from experiments import *
from strategies import *
from aggregation import *
from local_secrets import *
from local_logger import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from small_text.utils.annotations import ExperimentalWarning
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar
from huggingface_hub import login

import pathlib
from types import MethodType
import argparse


def ensure_interactive_plot_backend():
    backend_name = str(matplotlib.get_backend())
    backend_name_lower = backend_name.lower()
    needs_gui_backend = backend_name_lower == 'agg' or 'inline' in backend_name_lower

    if not needs_gui_backend:
        return

    for candidate_backend in ('MacOSX', 'QtAgg', 'TkAgg'):
        try:
            plt.switch_backend(candidate_backend)
            return
        except Exception:
            continue

    print(
        f'[WARN]: Could not switch matplotlib to a GUI backend (current backend: {backend_name}). '
        '3D rotation may be unavailable.'
    )


def main() -> int | None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--allow-llm',
        action='store_true',
        help='Enable LLM-backed strategies and API calls.',
    )
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=ExperimentalWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    disable_progress_bar()

    ensure_interactive_plot_backend()

    try:
        secrets = Secrets.load_secrets(pathlib.Path.cwd().parent)
        loaded_secrets = True
    except FileNotFoundError:
        print('Failed to load secrets')
        loaded_secrets = False

    if loaded_secrets:
        login(secrets.hf_token)

    logger = Logger(verbose_console=False, log_file=pathlib.Path.cwd().parent / "log.txt")

    if args.allow_llm:
        llms.enable()

    # with DataDatabase(pathlib.Path.cwd().parent, logger=logger) as db:
    #     db.try_restore()
    #     # db.recollect_stored()
    #     db.unroll()
        
    # return

    with DataDatabase(pathlib.Path.cwd().parent, logger=logger) as db:
        db.try_restore()
        ag_news = db.get_dataset(DatasetID('ag_news'), cache=True)
        sst_2 = db.get_dataset(DatasetID('glue', 'sst2'), text_field='sentence', cache=True)

        if loaded_secrets:
            db.connect_llm(llms.LLMType.GIGACHAT, llms.GigachatLLM.connect(secrets.gigachat_api_key))
        else:
            logger.warn(f'Failed to connect {llms.LLMType.GIGACHAT}')

        from_product = Experiments.from_product(
            database=db,
            datasets=[ag_news, sst_2],
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
                active_learning_strategies=[MockQueryStrategyType()],
                bugdets=[500],
                splits=list(range(10, 300 + 1, 10)),
                runs=5,
            ),
            from_product(
                active_learning_strategies=[QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE)],
                bugdets=[500],
                splits=list(range(10, 300 + 1, 10)),
                runs=3,
            ),
            # (
            #     Experiments.from_product(
            #         database=db,
            #         datasets=[sst_2, ag_news],
            #         seeds=[42, 69, 1337, 1147, 1984],
            #         pool_size=1000,
            #         cold_start_strategies=[SelectLLMQueryStrategyType(llms.LLMType.GIGACHAT, db)],
            #         active_learning_strategies=[QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE)],
            #         al_batch_size=10,
            #         bugdets=[500],
            #         splits=list(range(10, 300 + 1, 10)),
            #         runs=3,
            #     )
            #     if loaded_secrets
            #     else Experiments()
            # ),
            (
                Experiments.from_product(
                    database=db,
                    datasets=[ag_news, sst_2],
                    seeds=[42, 69, 1337, 1147, 1984],
                    pool_size=1000,
                    cold_start_strategies=[ActiveLLMQueryStrategyType(llms.LLMType.GIGACHAT, db)],
                    active_learning_strategies=[QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE)],
                    al_batch_size=10,
                    cs_batch_size=10,
                    bugdets=[500],
                    splits=list(range(10, 300 + 1, 10)),
                    runs=3,
                )
                if loaded_secrets
                else Experiments()
            ),
            # Experiments.from_product(
            #     database=db,
            #     datasets=[ag_news, sst_2],
            #     seeds=[42, 69, 1337, 1147, 1984],
            #     pool_size=1000,
            #     cold_start_strategies=[DiversityInitQueryStrategyType()],
            #     active_learning_strategies=[QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE)],
            #     al_batch_size=10,
            #     bugdets=[500],
            #     splits=list(range(10, 300 + 1, 10)),
            #     runs=3,
            # ),
        )

        experiments.sort_by(
            first=('seed', 'dataset'),
            last=({'attr': 'split', 'reverse': False}, {'attr': 'active_learning_strategy', 'reverse': True}),
        )

        tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

        def _encode_plus_adapter(self, text, text_pair=None, **kwargs):
            if text_pair is not None:
                return self(text, text_pair, **kwargs)
            return self(text, **kwargs)

        tokenizer.encode_plus = MethodType(_encode_plus_adapter, tokenizer)

        @dataclasses.dataclass
        class RunningInfo:
            total: int
            exps: list[int]
            at: int = 0
            order_at: int = 0

        info = RunningInfo(
            sum(
                exp not in db or db.retrieve(exp.get_id()).runs < experiments[exp][2].expected_runs
                for exp in experiments
            ),
            [
                experiments[exp][1]
                for exp in experiments
                if exp not in db or db.retrieve(exp.get_id()).runs < experiments[exp][2].expected_runs
            ],
        )

        def update_info(number: int, exp: Experiment, exp_info: Experiments.ExperimentInfo):
            info.at = info.exps.index(experiments[exp][1]) + 1
            info.order_at = number

        def log_run(exp: Experiment, run: int):
            logger.info(f'RUNNING EXPERIMENT [{info.at}#{info.order_at}({run})/{info.total}/{len(experiments)}]: {exp}')

        experiments.run_all(
            tokenizer,
            'google/bert_uncased_L-2_H-128_A-2',
            db,
            experiment_iteration_callback=update_info,
            run_iteration_callback=log_run,
            runs_in_depth=False,
        )

        db.recollect_stored()

        rng = ExperimentsRange(
            experiments,
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
                    dataclasses.replace(Experiments.ExperimentKey.empty(), split=split)
                    # for split in list(range(10, 50 + 1, 10))
                    for split in list(range(10, 300 + 1, 10))
                )
            ),
        )

        groups = ExperimentGroup.compose_groups(
            rng,
            [GroupEqFilter('cold_start_strategy.query_strategy'), GroupEqFilter('dataset.id'), GroupEqFilter('split')],
            [
                [],
                [],
                [],
                [
                    ComposeAggregator(agg, ZipAggregator(), MergeBundleAggregator(1), SpreadBundleAggregator(2))
                    for agg in (AverageAggregator(), VarianceAggregator(), PlainCountAggregator())
                ],
            ],  # TODO: think about redundant aggregators
            ExperimentSelector(
                [BundleAggregator()],
                ['final_accuracy', 'final_macro_f1'],
                runs=5,
            ),
        )

        # print(groups)
        # print(groups.contained[0].unique_tuples(("seed", "split")))
        printable = groups.to_printable()
        # aggregated = groups.aggregate()
        # print(printable)
        # print(groups_for_count.contained[0].aggregate())

        # return

        strategies = set()
        datasets = set()
        splits = set()
        data = {}

        for strat_key, inner in printable.items():
            strategy_obj = strat_key[0][1]
            strategy_name = str(strategy_obj)
            strategies.add(strategy_name)
            ds_dict = inner['printable']

            for ds_key, inner in ds_dict.items():
                dataset_id = ds_key[0][1]
                if dataset_id.subset:
                    ds_name = f'{dataset_id.path}_{dataset_id.subset}'
                else:
                    ds_name = dataset_id.path
                datasets.add(ds_name)
                split_dict = inner['printable']

                for split_key, values in split_dict.items():
                    split = split_key[0][1]
                    splits.add(split)
                    (mean_acc, mean_f1), (var_acc, var_f1), (count,) = values['printable']
                    data[(strategy_name, ds_name, split)] = (mean_acc, mean_f1, var_acc, var_f1, count)

        datasets = sorted(datasets)
        strategies = sorted(strategies)
        splits_sorted = sorted(splits)

        metric_configs = [
            (0, 'Final Accuracy', 0, 2),  # mean at index 0, variance at index 2
            (1, 'Final Macro F1', 1, 3),  # mean at index 1, variance at index 3
        ]
        color_cycle = (
            plt.rcParams['axes.prop_cycle']
            .by_key()
            .get(
                'color',
                ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
            )
        )
        strategy_colors = {strategy: color_cycle[idx % len(color_cycle)] for idx, strategy in enumerate(strategies)}

        fig, axes = plt.subplots(2, len(datasets), figsize=(8 * len(datasets), 12), sharex=True, squeeze=False)

        for row, metric_name, metric_idx, var_idx in metric_configs:
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
                    ax.errorbar(
                        splits_sorted,
                        y_mean,
                        yerr=y_std,
                        marker='o',
                        label=strategy,
                        capsize=3,
                        color=strategy_colors[strategy],
                    )
                ax.set_title(f'{ds} – {metric_name}')
                ax.set_xlabel('Split')
                ax.set_ylabel(metric_name)
                ax.legend()
                ax.grid(True)

        plt.tight_layout()
        if hasattr(fig.canvas.manager, 'set_window_title'):
            fig.canvas.manager.set_window_title('2D Metrics')
        plt.show()

        fig_3d = plt.figure(figsize=(8 * len(datasets), 12))

        for row, metric_name, metric_idx, var_idx in metric_configs:
            for col, ds in enumerate(datasets):
                subplot_index = row * len(datasets) + col + 1
                ax = fig_3d.add_subplot(2, len(datasets), subplot_index, projection='3d')
                ax.set_proj_type('persp')

                for strategy_idx, strategy in enumerate(strategies):
                    z_mean = []
                    z_std = []
                    for split in splits_sorted:
                        vals = data.get((strategy, ds, split))
                        if vals is not None:
                            z_mean.append(vals[metric_idx])
                            z_std.append(np.sqrt(vals[var_idx]))
                        else:
                            z_mean.append(np.nan)
                            z_std.append(np.nan)

                    x_values = np.asarray(splits_sorted, dtype=float)
                    z_mean_values = np.asarray(z_mean, dtype=float)
                    z_std_values = np.asarray(z_std, dtype=float)
                    valid_mask = ~(np.isnan(z_mean_values) | np.isnan(z_std_values))

                    if not np.any(valid_mask):
                        continue

                    x_valid = x_values[valid_mask]
                    z_mean_valid = z_mean_values[valid_mask]
                    z_std_valid = z_std_values[valid_mask]
                    y_valid = np.full(x_valid.shape, strategy_idx, dtype=float)
                    lower = z_mean_valid - z_std_valid
                    upper = z_mean_valid + z_std_valid
                    color = strategy_colors[strategy]

                    ax.plot(x_valid, y_valid, z_mean_valid, marker='o', color=color, linewidth=2)
                    ax.plot(x_valid, y_valid, lower, color=color, alpha=0.6, linewidth=1.1, linestyle='--')
                    ax.plot(x_valid, y_valid, upper, color=color, alpha=0.6, linewidth=1.1, linestyle='--')

                    for x_value, lower_value, upper_value in zip(x_valid, lower, upper):
                        ax.plot(
                            [x_value, x_value],
                            [strategy_idx, strategy_idx],
                            [lower_value, upper_value],
                            color=color,
                            alpha=0.45,
                            linewidth=1.2,
                        )

                    band_vertices = [
                        *zip(x_valid, y_valid, lower),
                        *zip(x_valid[::-1], y_valid[::-1], upper[::-1]),
                    ]
                    ax.add_collection3d(
                        Poly3DCollection(
                            [band_vertices],
                            facecolors=color,
                            edgecolors=color,
                            linewidths=1.0,
                            alpha=0.18,
                        )
                    )

                ax.set_title(f'{ds} – {metric_name} (3D)')
                ax.set_xlabel('Split')
                ax.set_ylabel('Strategy')
                ax.set_zlabel(metric_name)
                ax.set_yticks(np.arange(len(strategies)))
                ax.set_yticklabels(strategies)
                ax.view_init(elev=24, azim=-62)
                ax.grid(True)

        plt.tight_layout()
        if hasattr(fig_3d.canvas.manager, 'set_window_title'):
            fig_3d.canvas.manager.set_window_title('3D Metrics (drag to rotate)')
        plt.show()


if __name__ == '__main__':
    raise SystemExit(main())
