from database import *
from experiments import *
from strategies import *
from aggregation import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from small_text.utils.annotations import ExperimentalWarning
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

import pathlib
from types import MethodType


def main() -> int | None:
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    disable_progress_bar()

    def ensure_interactive_plot_backend():
        backend_name = str(matplotlib.get_backend())
        backend_name_lower = backend_name.lower()
        needs_gui_backend = backend_name_lower == "agg" or "inline" in backend_name_lower

        if not needs_gui_backend:
            return

        for candidate_backend in ("MacOSX", "QtAgg", "TkAgg"):
            try:
                plt.switch_backend(candidate_backend)
                return
            except Exception:
                continue

        print(
            f"[WARN]: Could not switch matplotlib to a GUI backend (current backend: {backend_name}). "
            "3D rotation may be unavailable."
        )

    ensure_interactive_plot_backend()

    with DataDatabase(pathlib.Path.cwd().parent) as db:
        # db.try_restore()
        ag_news = db.get_dataset(DatasetID("ag_news"))
        sst_2 = db.get_dataset(DatasetID("glue", "sst2"), text_field="sentence")

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
            # from_product(
            #     active_learning_strategies=[QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE)],
            #     bugdets=[500],
            #     splits=list(range(10, 300 + 1, 10)),
            # ),
        )

        experiments.sort_by(first=("seed", "dataset"), last=({"attr": "split", "reverse": False},))

        tokenizer = AutoTokenizer.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2", cache_dir=pathlib.Path("./cache")
        )

        def _encode_plus_adapter(self, text, text_pair=None, **kwargs):
            if text_pair is not None:
                return self(text, text_pair, **kwargs)
            return self(text, **kwargs)

        tokenizer.encode_plus = MethodType(_encode_plus_adapter, tokenizer)

        runs = 3

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
                f"[INFO]: RUNNING EXPERIMENT [{info.at}#{info.order_at}({run})/{info.total}/{len(experiments)}]: {exp}"
            )

        experiments.run_all(
            tokenizer,
            "google/bert_uncased_L-2_H-128_A-2",
            runs,
            db,
            experiment_iteration_callback=update_info,
            run_iteration_callback=log_run,
        )

        # db.recollect_stored()

        rng = ExperimentsRange(
            experiments,
            ExperimentFilter(
                (
                    ExperimentKeyExpression(
                        dataclasses.replace(Experiments.ExperimentKey.empty(), dataset_id=DatasetID("ag_news"))
                    )
                    | ExperimentKeyExpression(
                        dataclasses.replace(Experiments.ExperimentKey.empty(), dataset_id=DatasetID("glue", "sst2"))
                    )
                )
                & ExperimentKeyExpression(
                    dataclasses.replace(Experiments.ExperimentKey.empty(), al_strategy=MockQueryStrategyType())
                )
                & ExperimentInExpression(
                    dataclasses.replace(Experiments.ExperimentKey.empty(), split=split)
                    for split in list(range(10, 300 + 1, 10))
                    # for split in list(range(10, 50 + 1, 10))
                )
            ),
        )

        groups = ExperimentGroup.compose_groups(
            rng,
            [GroupEqFilter("cold_start_strategy.query_strategy"), GroupEqFilter("dataset.id"), GroupEqFilter("split")],
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
                ["final_accuracy", "final_macro_f1"],
                runs=5,
            ),
            # ExperimentSelector(
            #     [BundleAggregator()],
            #     ["final_accuracy", "final_macro_f1"],
            #     runs=5,
            # ),
        )
        groups_for_count = ExperimentGroup.compose_groups(
            rng,
            [GroupEqFilter("cold_start_strategy.query_strategy"), GroupEqFilter("dataset.id"), GroupEqFilter("split")],
            [[], [], [], [ComposeAggregator(PlainCountAggregator(), SpreadBundleAggregator(3))]],
            ExperimentSelector(
                [SpreadBundleAggregator(1)],
                ["uuid"],
                runs=5,
            ),
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

        for strat_key, inner in printable.items():
            strategy_obj = strat_key[0][1]
            strategy_name = str(strategy_obj)
            strategies.add(strategy_name)
            ds_dict = inner["printable"]

            for ds_key, inner in ds_dict.items():
                dataset_id = ds_key[0][1]
                if dataset_id.subset:
                    ds_name = f"{dataset_id.path}_{dataset_id.subset}"
                else:
                    ds_name = dataset_id.path
                datasets.add(ds_name)
                split_dict = inner["printable"]

                for split_key, values in split_dict.items():
                    split = split_key[0][1]
                    splits.add(split)
                    (mean_acc, mean_f1), (var_acc, var_f1), (count,) = values["printable"]
                    # acc_per_run, f1_per_run = values[0][0]
                    # data[(strategy_name, ds_name, split)] = (acc_per_run, f1_per_run)
                    data[(strategy_name, ds_name, split)] = (mean_acc, mean_f1, var_acc, var_f1, count)

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

        metric_configs = [
            (0, "Final Accuracy", 0, 2),  # mean at index 0, variance at index 2
            (1, "Final Macro F1", 1, 3),  # mean at index 1, variance at index 3
        ]
        color_cycle = (
            plt.rcParams["axes.prop_cycle"]
            .by_key()
            .get(
                "color",
                ["tab:blue", "tab:orange", "tab:green", "tab:red"],
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
                        marker="o",
                        label=strategy,
                        capsize=3,
                        color=strategy_colors[strategy],
                    )
                ax.set_title(f"{ds} – {metric_name}")
                ax.set_xlabel("Split")
                ax.set_ylabel(metric_name)
                ax.legend()
                ax.grid(True)

        plt.tight_layout()
        if hasattr(fig.canvas.manager, "set_window_title"):
            fig.canvas.manager.set_window_title("2D Metrics")
        plt.show()

        fig_3d = plt.figure(figsize=(8 * len(datasets), 12))

        for row, metric_name, metric_idx, var_idx in metric_configs:
            for col, ds in enumerate(datasets):
                subplot_index = row * len(datasets) + col + 1
                ax = fig_3d.add_subplot(2, len(datasets), subplot_index, projection="3d")
                ax.set_proj_type("persp")

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

                    ax.plot(x_valid, y_valid, z_mean_valid, marker="o", color=color, linewidth=2)
                    ax.plot(x_valid, y_valid, lower, color=color, alpha=0.6, linewidth=1.1, linestyle="--")
                    ax.plot(x_valid, y_valid, upper, color=color, alpha=0.6, linewidth=1.1, linestyle="--")

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

                ax.set_title(f"{ds} – {metric_name} (3D)")
                ax.set_xlabel("Split")
                ax.set_ylabel("Strategy")
                ax.set_zlabel(metric_name)
                ax.set_yticks(np.arange(len(strategies)))
                ax.set_yticklabels(strategies)
                ax.view_init(elev=24, azim=-62)
                ax.grid(True)

        plt.tight_layout()
        if hasattr(fig_3d.canvas.manager, "set_window_title"):
            fig_3d.canvas.manager.set_window_title("3D Metrics (drag to rotate)")
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
