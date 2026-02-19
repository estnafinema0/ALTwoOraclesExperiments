from database import *
from experiments import *
from strategies import *

import warnings
from small_text.utils.annotations import ExperimentalWarning
from transformers import AutoTokenizer


def main() -> int | None:
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    with DataDatabase(".") as db:
        db.recollect_stored()
        ag_news = db.get_dataset(DatasetID("ag_news"))
        sst_2 = db.get_dataset(DatasetID("glue", "sst2"), text_field="sentence")
        experiments = Experiments.from_experiments(
            Experiments.from_product(
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
                active_learning_strategies=[MockQueryStrategyType()],
                bugdets=[500],
                splits=list(range(10, 300 + 1, 10)),
                al_batch_size=10,  # currently mock strategy
            )
        )

        experiments.run_all(
            AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2"),
            "google/bert_uncased_L-2_H-128_A-2",
            5,
            db,
        )
        db.recollect_stored()


if __name__ == "__main__":
    raise SystemExit(main())
