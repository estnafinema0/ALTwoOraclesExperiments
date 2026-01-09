import itertools 
from transformers import AutoTokenizer
from definitions import *

def main():
    init(c)

    raw_datasets: dict[str, Dataset] = {}
    standardized_datasets: dict[str, Dataset] = {}

    for key, cfg in c.DATASETS.items():
        print(f'Loading Dataset {key}')
        ds = load_dataset_by_key(cfg)
        ds_std = standardize_dataset(ds, cfg)
        raw_datasets[key] = ds
        standardized_datasets[key] = ds_std

    for key, ds_std in standardized_datasets.items():
        dataset_basic_stats(key, ds_std)
    
    tokenizer = AutoTokenizer.from_pretrained(c.TRANSFORMER_MODEL_NAME)
    
    histories_sst2 = run_all_seeds_and_strategies_for_dataset(
        tokenizer,
        dataset_key="sst2",
        standardized_datasets=standardized_datasets,
        cfg=ALConfig(),
        seeds=[42, 68, 1337, 1147, 1984]
    )

# тут были комментарии


    # summaries_sst2 = {
    #     "Random": [value for key, value in histories_sst2.items() if key[0] == QueryStrategy.RANDOM],
    #     "LeastConf": [value for key, value in histories_sst2.items() if key[0] == QueryStrategy.LEAST_CONFIDENCE],
    #     "BALD": [value for key, value in histories_sst2.items() if key[0] == QueryStrategy.BALD],
    #     "BADGE": [value for key, value in histories_sst2.items() if key[0] == QueryStrategy.BADGE],
    # }

    # summaries_sst2 = {key : summarize_cold_start(value) for key, value in summaries_sst2.items()}
    # for name, data in summaries_sst2.items():
    #     xs = sorted(data.keys())
    #     ys = [data[k]["acc"] for k in xs]
    #     plt.plot(xs, ys, marker='o', label=name)

    # by_seed_sst2 = [key + (value,) for key, value in histories_sst2.items() if key[1] == 42]
    # for strat, _, hist in by_seed_sst2:
    #     plot_single_history(pd.DataFrame(hist), f'SST2 Стратегия {strat}')


    # plt.title("SST-2: Accuracy vs Number of Labeled Samples")
    # plt.xlabel("Number of labeled samples")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # from transformers import DistilBertForSequenceClassification

    # if not hasattr(DistilBertForSequenceClassification, "_orig_forward"):
    #     DistilBertForSequenceClassification._orig_forward = DistilBertForSequenceClassification.forward

    # def _forward_patched(self, input_ids=None, attention_mask=None, **kwargs):
    #     return DistilBertForSequenceClassification._orig_forward(
    #         self,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         **{k: v for k, v in kwargs.items() if k != "token_type_ids"}
    #     )

    # DistilBertForSequenceClassification.forward = _forward_patched

    # print("Patched DistilBertForSequenceClassification.forward (idempotent).")






    # histories_ag_news = run_all_seeds_and_strategies_for_dataset(
    #     tokenizer,
    #     dataset_key="ag_news",
    #     standardized_datasets=standardized_datasets,
    #     cfg=ALConfig(),
    #     seeds=[42, 68, 1337, 1147, 1984]
    # )


    # summaries_ag_news = {
    #     "Random": [value for key, value in histories_ag_news.items() if key[0] == QueryStrategy.RANDOM],
    #     "LeastConf": [value for key, value in histories_ag_news.items() if key[0] == QueryStrategy.LEAST_CONFIDENCE],
    #     "BALD": [value for key, value in histories_ag_news.items() if key[0] == QueryStrategy.BALD],
    #     "BADGE": [value for key, value in histories_ag_news.items() if key[0] == QueryStrategy.BADGE],
    # }
    # summaries_ag_news = {key : summarize_cold_start(value) for key, value in summaries_ag_news.items()}

    
    # by_seed_ag_news = [key + (value,) for key, value in histories_ag_news.items() if key[1] == 42]
    # for strat, _, hist in by_seed_ag_news:
    #     plot_single_history(pd.DataFrame(hist), f'AG News Стратегия {strat}')

    # plt.figure(figsize=(8,5))
    # for name, data in summaries_ag_news.items():
    #     xs = sorted(data.keys())
    #     ys = [data[k]["acc"] for k in xs]
    #     plt.plot(xs, ys, marker="o", label=name)

    # plt.xlabel("Число размеченных объектов")
    # plt.ylabel("Accuracy (validation)")
    # plt.title("AG News: Accuracy vs #labeled для четырёх стратегий")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # 
    client = GigaChat(
        credentials=c.GIGACHAT_CREDENTIALS,
        model=c.GIGACHAT_MODEL,
        verify_ssl_certs=False
    )
    oracle_strategies = [
        OracleStrategy(OracleType.LLM, gigachat_client=client),
        OracleStrategy(OracleType.HYBRID, hybrid_k=3, gigachat_client=client),
        OracleStrategy(OracleType.HYBRID, hybrid_k=10, gigachat_client=client),
        # OracleStrategy(OracleType.HYBRID, hybrid_k=20, gigachat_client=client),
    ]

    oracle_historiess = {}
    for oracle_strategy, dataset in itertools.product(oracle_strategies, ['sst2', 'ag_news']):
        oracle_historiess[(oracle_strategy, dataset)] = run_all_strategies_for_dataset(tokenizer, dataset, standardized_datasets, ALConfig(), oracle_strategy)
    exit(0)
        
   
    # history_sst2_random_diversity_human = run_al_experiment_diversity_init(
    #     strategy_name="random",
    #     cfg=AL_CFG_SST2_LLM,
    #     experiment_name="sst2_random_diversityinit_human",
    #     oracle_type="human",
    #     dataset_key="sst2",
    #     hybrid_k=None,
    #     gigachat_client=None,
    #     initial_oracle_type="human",
    # )

    # history_sst2_lc_diversity_human = run_al_experiment_sst2_diversity_init(
    #     strategy_name="least_conf",
    #     cfg=AL_CFG_SST2_LLM,
    #     experiment_name="sst2_leastconf_diversityinit_human",
    # )

    # history_sst2_bald_diversity_human = run_al_experiment_sst2_diversity_init(
    #     strategy_name="bald",
    #     cfg=AL_CFG_SST2_LLM,
    #     experiment_name="sst2_bald_diversityinit_human",
    # )

    # history_sst2_random_diversity_llm = run_al_experiment_diversity_init(
    #     strategy_name="random",
    #     cfg=AL_CFG_SST2_LLM,
    #     experiment_name="sst2_random_diversityinit_llm",
    #     oracle_type="llm",
    #     dataset_key="sst2",
    #     hybrid_k=None,
    #     gigachat_client=client,
    #     initial_oracle_type="llm",   # стартовые 200 тоже размечает LLM
    # )

    # history_sst2_random_diversity_hybrid_k3 = run_al_experiment_diversity_init(
    #     strategy_name="random",
    #     cfg=AL_CFG_SST2_LLM,
    #     experiment_name="sst2_random_diversityinit_hybrid_k3",
    #     oracle_type="hybrid",
    #     dataset_key="sst2",
    #     hybrid_k=3,                   # 3 итерации LLM, потом human
    #     gigachat_client=client,
    #     initial_oracle_type="human",  # например, стартовые 200 считаем человеческими
    # )


    # plot_diversity_init_comparison_sst2()

    plot_oracle_comparison_sst2(
        strategy_name="random",
        human_exp="sst2_random",                 # human-only базовый эксперимент
        llm_exp="sst2_random_llm_only",
        hybrid_exp="sst2_random_hybrid_k3",
        title_prefix="SST-2, Random"
    )

    plot_oracle_comparison_sst2(
        strategy_name="least_conf",
        human_exp="sst2_least_conf",
        llm_exp="sst2_leastconf_llm_only",
        hybrid_exp="sst2_leastconf_hybrid_k3",
        title_prefix="SST-2, Least Confidence"
    )

    plot_oracle_comparison_sst2(
        strategy_name="bald",
        human_exp="sst2_bald",
        llm_exp="sst2_bald_llm_only",
        hybrid_exp="sst2_bald_hybrid_k3",
        title_prefix="SST-2, BALD"
    )

    raw_datasets = {}
    standardized_datasets = {}

    for key, cfg in DATASETS.items():
        ds = load_dataset_by_key(key, cfg)
        raw_datasets[key] = ds
        ds_std = standardize_dataset(ds, cfg, key)
        standardized_datasets[key] = ds_std
        dataset_basic_stats(key, ds_std)

    emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    emb_model = AutoModel.from_pretrained(EMB_MODEL_NAME).to(device)
    emb_model.eval()


# if __name__ == "__main__":
#     # Применяем патч для DistilBERT если используется
#     try:
#         from transformers import DistilBertForSequenceClassification
#         if not hasattr(DistilBertForSequenceClassification, "_orig_forward"):
#             DistilBertForSequenceClassification._orig_forward = DistilBertForSequenceClassification.forward

#         def _forward_patched(self, input_ids=None, attention_mask=None, **kwargs):
#             return DistilBertForSequenceClassification._orig_forward(
#                 self,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **{k: v for k, v in kwargs.items() if k != "token_type_ids"}
#             )

#         DistilBertForSequenceClassification.forward = _forward_patched
#         print("Applied patch for DistilBertForSequenceClassification.forward")
#     except:
#         pass

#     # Запуск основных экспериментов
#     main()

# def main():
#     """Основная функция для запуска всех экспериментов"""

#     # Создаем конфигурацию с нужным диапазоном
#     cfg = ALConfig(
#         seed=42,
#         pool_size=POOL_SIZE,
#         initial_labeled=INITIAL_LABELED,
#         batch_size=BATCH_SIZE,
#         iterations=N_ITER,
#         max_labeled=MAX_LABELED,
#         cost_human=COST_HUMAN,
#         cost_llm=COST_LLM,
#         max_length=MAX_LENGTH,
#         num_epochs=3,
#         train_batch_size=16,
#     )

#     print("\n" + "="*80)
#     print("ACTIVE LEARNING EXPERIMENTS WITH CACHING")
#     print("="*80)
#     print(f"Model: {TRANSFORMER_MODEL_NAME}")
#     print(f"Number of runs per experiment: {NUM_RUNS}")
#     print(f"Labeled samples range: {INITIAL_LABELED} to {MAX_LABELED} (step {BATCH_SIZE})")
#     print(f"Seeds: {SEEDS[:NUM_RUNS]}")
#     print(f"Log directory: {LOG_DIR}")
#     print(f"Aggregated log directory: {AGGREGATED_LOG_DIR}")
#     print("="*80 + "\n")

#     # Настройка глобального логирования
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(PROJECT_ROOT, 'experiment_main.log'), encoding='utf-8'),
#             logging.StreamHandler()
#         ]
#     )

#     logger = logging.getLogger(__name__)
#     logger.info("Starting main experiment execution")

#     # Список стратегий для тестирования
#     strategies = ["random", "least_conf", "bald", "badge"]

#     # Параметры кэширования
#     USE_CACHE = True
#     FORCE_RERUN = False  # Принудительный перезапуск всех экспериментов

#     all_datasets_results = {}

#     # Запуск экспериментов для каждого датасета
#     for dataset_key in DATASETS.keys():
#         print(f"\n{'#'*80}")
#         print(f"EXPERIMENTS FOR DATASET: {dataset_key.upper()}")
#         print(f"Use cache: {USE_CACHE}")
#         print(f"Force rerun: {FORCE_RERUN}")
#         print('#'*80)

#         # Логирование информации о датасете
#         if 'log_dataset_statistics' in globals():
#             log_dataset_statistics(dataset_key)

#         # Сравнение стратегий с кэшированием
#         results = compare_strategies(
#             dataset_key=dataset_key,
#             strategies=strategies,
#             seeds=SEEDS[:NUM_RUNS],
#             cfg=cfg,
#             use_cache=USE_CACHE,
#             force_rerun=FORCE_RERUN
#         )

#         all_datasets_results[dataset_key] = results

#         # Сохранение сводной статистики
#         save_summary_statistics(dataset_key, results)

#         # Визуализация сравнения всех стратегий
#         if 'plot_strategy_comparison' in globals() and results:
#             plot_strategy_comparison(dataset_key, results)

#         logger.info(f"Completed experiments for dataset: {dataset_key}")

#     # Генерация итогового отчета
#     if 'generate_final_report' in globals():
#         generate_final_report(all_datasets_results, cfg)

#     print("\n" + "="*80)
#     print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
#     print(f"Results saved in: {AGGREGATED_LOG_DIR}")
#     print("="*80 + "\n")

#     logger.info("All experiments completed successfully")

# if __name__ == "__main__":
#     # Применяем патч для DistilBERT если используется
#     try:
#         from transformers import DistilBertForSequenceClassification
#         if not hasattr(DistilBertForSequenceClassification, "_orig_forward"):
#             DistilBertForSequenceClassification._orig_forward = DistilBertForSequenceClassification.forward

#         def _forward_patched(self, input_ids=None, attention_mask=None, **kwargs):
#             return DistilBertForSequenceClassification._orig_forward(
#                 self,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **{k: v for k, v in kwargs.items() if k != "token_type_ids"}
#             )

#         DistilBertForSequenceClassification.forward = _forward_patched
#         print("Applied patch for DistilBertForSequenceClassification.forward")
#     except:
#         pass

#     # Запуск основных экспериментов
#     main()
#     client = GigaChat(
#         credentials='MDE5YTg3MDEtMjBjYy03YzJlLTg3ZDQtMTYxNGYxNmYwMDc1OjM4ZmM3NWQwLWMxMjYtNGM4Yy1hZTlmLWNmZjI1Mzc1MDU1OA==',
#         scope=SCOPE,
#         model="GigaChat",
#         verify_ssl_certs=False
#     )

if __name__ == '__main__':
    main()


# try:
#     emb_tokenizer = AutoTokenizer.from_pretrained(DIV_CFG.emb_model_name)
#     emb_model = AutoModel.from_pretrained(DIV_CFG.emb_model_name).to(DEVICE)
#     emb_model.eval()
#     print(f"Embedding model loaded: {DIV_CFG.emb_model_name}")
# except Exception as e:
#     print(f"Failed to load embedding model: {e}")
#     emb_model = None

# if __name__ == "__main__":
#     # Запуск diversity экспериментов
#     run_diversity_experiments_main()
