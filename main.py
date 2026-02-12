from experiments import *
from database import *
from strategies import *

from transformers import AutoTokenizer


def main() -> int | None:
    with DataDatabase(".") as db:
        ag_news = db.get(DatasetID("ag_news"))
        sst_2 = db.get(DatasetID("glue", "sst2"), text_field="sentence")
        experiments = Experiments.from_experiments(
            Experiments.from_product(
                datasets=[ag_news, sst_2],
                seeds=[
                    # 42, 
                       69, 
                    #    1337, 1147, 1984
                    ],
                pool_size=1000,
                cold_start_strategies=[
                    QueryStrategySimple(SimpleQueryStrategyType.RANDOM),
                    QueryStrategySimple(SimpleQueryStrategyType.LEAST_CONFIDENCE),
                    QueryStrategySimple(SimpleQueryStrategyType.BALD),
                    QueryStrategySimple(SimpleQueryStrategyType.BADGE),
                ],
                active_learning_strategies=[MockQueryStrategyType()],
                bugdets=[500],
                # splits=list(range(10, 300 + 1, 10)),
                splits=[10],
                al_batch_size=10,  # currently mock strategy
            )
        )

        experiments.run_all(
            AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2"),
            "google/bert_uncased_L-2_H-128_A-2",
            1,
            db
        )


if __name__ == "__main__":
    raise SystemExit(main())
