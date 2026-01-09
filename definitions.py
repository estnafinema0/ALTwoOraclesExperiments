from collections import Counter
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from datetime import datetime
from gigachat import GigaChat
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from small_text import PoolBasedActiveLearner, random_initialization_balanced, TransformerBasedClassificationFactory, TransformerModelArguments, TransformersDataset
from small_text.integrations.pytorch.query_strategies import BADGE
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.query_strategies.bayesian import BALD
from small_text.query_strategies.strategies import RandomSampling, LeastConfidence
from tqdm.auto import tqdm
from transformers import DistilBertForSequenceClassification, AutoTokenizer, AutoModel, BertTokenizer, BertModel
from typing import Any, Optional, List, Dict, Tuple
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import enum
import dataclasses 

class QueryStrategy(enum.Enum):
    RANDOM = enum.auto()
    LEAST_CONFIDENCE = enum.auto()
    BALD = enum.auto()
    BADGE = enum.auto()
    
class OracleType(enum.Enum):
    HUMAN = enum.auto()
    LLM = enum.auto()
    HYBRID = enum.auto()
    
@dataclass
class DatasetConfig:
    hf_name: str
    hf_config: str | None
    text_field: str
    label_field: str

class Config:
    SEED = 42
    PROJECT_ROOT = os.getcwd()
    COST_HUMAN = 5.0
    COST_LLM = 1.0
    # PROJECT_ROOT = "/content/drive/MyDrive/al_two_oracles2/"
    DATASETS = {
        "sst2": DatasetConfig(
            hf_name="glue",
            hf_config="sst2",
            text_field="sentence",
            label_field="label",
        ),
        "ag_news": DatasetConfig(
            hf_name="ag_news",
            hf_config=None,
            text_field="text",
            label_field="label",
        ),
        # "hatexplain": DatasetConfig(
        #     hf_name="hatexplain",
        #     hf_config=None,
        #     text_field="post",
        #     label_field="annotated_label"
        # ),
    }
    POOL_SIZE = 1000
    INITIAL_LABELED = 10
    BATCH_SIZE = 10
    N_ITER = (300 - INITIAL_LABELED) // BATCH_SIZE + 1
    TRANSFORMER_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
    MAX_LENGTH = 128
    AL_CFG_SST2_LLM: 'ALConfig'
    EMB_MODEL_NAME = TRANSFORMER_MODEL_NAME
    GIGACHAT_AUTH_KEY = "GIGACHAT_AUTH_KEY"
    SCOPE = "GIGACHAT_API_PERS"
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models_checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    AGGREGATED_LOG_DIR = os.path.join(PROJECT_ROOT, "aggregated_logs")
    LABEL_SPACES = {
        "sst2": {
            "id2label": {0: "negative", 1: "positive"},
            "label2id": {"negative": 0, "positive": 1},
            "task_description": "Классификация тональности английских предложений."
        },
        "ag_news": {
            "id2label": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "label2id": {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
            "task_description": "Классификация новостных заголовков по тематическим рубрикам."
        },
    }
    DEVICE: str
    GIGACHAT_CREDENTIALS = "=="
    GIGACHAT_MODEL = "GigaChat"
    EMB_TOKENIZER: BertTokenizer
    EMB_MODEL: BertModel

c = Config()

@dataclass
class OracleStrategy:
    type: OracleType
    hybrid_k: int | None = None
    gigachat_client: GigaChat | None = None

    def __init__(self, type, hybrid_k = None, gigachat_client = None):
        self.type = type
        if self.type == OracleType.HYBRID and (gigachat_client is None or hybrid_k is None):
            raise ValueError('If OracleType is Hybrid gigachat_client and hybrid_k must be specified')
        self.gigachat_client = gigachat_client
        self.hybrid_k = hybrid_k

    def to_real_oracle_type(self, current_k) -> OracleType:
        if self.type == OracleType.HYBRID and self.hybrid_k < current_k:
            return OracleType.HUMAN
        elif self.type == OracleType.HYBRID or self.type == OracleType.LLM:
            return OracleType.LLM
        else:
            return OracleType.HUMAN
        
@dataclass
class ALConfig:
    seed: int = c.SEED
    pool_size: int = c.POOL_SIZE
    initial_labeled: int = c.INITIAL_LABELED
    batch_size: int = c.BATCH_SIZE
    iterations: int = c.N_ITER
    cost_human: float = c.COST_HUMAN
    cost_llm: float = c.COST_LLM
    max_length: int = c.MAX_LENGTH
    num_epochs: int = 3
    train_batch_size: int = c.BATCH_SIZE
    force_cache: bool = True

# c.AL_CFG_SST2_LLM = ALConfig(
#     seed=42,
#     pool_size=5000,      # вместо 10000
#     initial_labeled=200,
#     batch_size=200,
#     iterations=5,        # 0..5 -> 200, 400, 600, 800, 1000, 1200
#     cost_human=5.0,
#     cost_llm=1.0,
#     max_length=128,
#     num_epochs=3,
#     train_batch_size=16,
# )


@dataclass
class DiversityConfig:
    use_diversity_init: bool = True
    diversity_samples: int = 200 # Количество семплов для diversity инициализации
    oracle_type: str = "hybrid" # "human", "llm", "hybrid"
    hybrid_k: int = 3 # Количество итераций с LLM в гибридном режиме
    use_cached_llm_labels: bool = True # Использовать кэшированные LLM метки
    llm_cache_dir: str = os.path.join(c.PROJECT_ROOT, "llm_cache")

    # Параметры для эмбеддингов
    emb_model_name: str = c.TRANSFORMER_MODEL_NAME
    emb_batch_size: int = 32
    emb_max_length: int = c.MAX_LENGTH

c.DIV_CFG = DiversityConfig()

def init(c: Config):
    os.makedirs(c.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(c.LOG_DIR, exist_ok=True)
    os.makedirs(c.DIV_CFG.llm_cache_dir, exist_ok=True)
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c.SEED)

    c.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    c.EMB_TOKENIZER = AutoTokenizer.from_pretrained(c.EMB_MODEL_NAME)
    c.EMB_MODEL = AutoModel.from_pretrained(c.EMB_MODEL_NAME).to(c.DEVICE)
    c.EMB_MODEL.eval()

def load_dataset_by_key(cfg) -> Dataset:
    if cfg.hf_config is None:
        return load_dataset(cfg.hf_name, trust_remote_code=True)
    else:
        return load_dataset(cfg.hf_name, cfg.hf_config, trust_remote_code=True)

def standardize_dataset(ds, cfg):
    text_field = cfg.text_field
    label_field = cfg.label_field

    return ds.map(
        lambda ex: {
            "text": ex[text_field],
            "label": ex[label_field],
        },
        remove_columns=ds["train"].column_names,
    )

def dataset_basic_stats(key, ds):
    print(f"\n===== BASIC STATS: {key.upper()} =====")
    splits = list(ds.keys())
    print("Splits:", splits)

    train = ds["train"]
    valid_split_name = "validation" if "validation" in ds else "val" if "val" in ds else None
    test_split_name = "test" if "test" in ds else None

    print("Train size:", len(train))
    if valid_split_name:
        print("Validation size:", len(ds[valid_split_name]))
    if test_split_name:
        print("Test size:", len(ds[test_split_name]))

    labels = [ex["label"] for ex in train]
    label_counts = Counter(labels)
    print("Label distribution (train):", label_counts)


def summarize_cold_start(histories, points=(10, 20, 30, 50, 100, 200, 300)):
    """
    history: список словарей из run_al_experiment_one.
    points: интересующие значения labeled.
    Возвращает словарь: {labeled: (acc, macro_f1)}.
    """
    result = {}
    for n in points:
        agg = ([], [])
        for history in histories:
            df = pd.DataFrame(history)
            
            row = df[df["labeled"] == n]
            if not row.empty:
                r = row.iloc[0]
                agg[0].append(float(r["acc"]))
                agg[1].append(float(r["macro_f1"]))

        if agg[0]:
            result[int(n)] = dict(acc=sum(agg[0])/len(agg[0]), macro_f1=sum(agg[1])/len(agg[1]))
        else:
            result[int(n)] = None
    return result 

def get_num_labels_for_dataset(key):
    return {'sst2': 2, 'ag_news': 4}.get(key)


def tokenize_dataset(ds, key):
    """
    Токенизируем датасет для sst2 / ag_news.
    """

    def _tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=c.MAX_LENGTH,
        )

    print(f"\nTokenizing dataset: {key}")
    tokenized = ds.map(_tokenize_batch, batched=True)

    tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return tokenized

def train_simple_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_simple(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def save_active_learner_checkpoint(tokenizer, active_learner, experiment_name: str, iteration: int):
    """
    Сохраняем текущую модель (и токенизатор) на диск.
    experiment_name: строка-идентификатор эксперимента, например 'sst2_random' или 'sst2_lc'.
    iteration: номер итерации AL (0..T).
    """
    # Папка вида: models_checkpoints/sst2_random/iter_03
    exp_dir = os.path.join(c.CHECKPOINT_DIR, experiment_name, f"iter_{iteration:02d}")
    os.makedirs(exp_dir, exist_ok=True)

    model = getattr(active_learner.classifier, "model", None)
    if model is None:
        print("[WARN] Не удалось найти .model у active_learner.classifier, пропускаю сохранение модели.")
        return

    print(f"Saving checkpoint to: {exp_dir}")
    model.save_pretrained(exp_dir)
    tokenizer.save_pretrained(exp_dir)


def save_history(history, experiment_name: str):
    """
    Сохраняет список словарей history в JSON и CSV в LOG_DIR.
    """
    os.makedirs(c.LOG_DIR, exist_ok=True)
    csv_path = os.path.join(c.LOG_DIR, f"{experiment_name}_history.csv")

    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    print(f"History saved to: {csv_path}")


def to_transformers_dataset(tokenizer, hf_split, num_classes, max_length=128):
    """
    hf_split: HuggingFace Dataset(split) с полями 'text' и 'label'.
    num_classes: количество классов.
    max_length: максимальная длина последовательности для токенизации.

    Возвращает small_text.TransformersDataset
    """
    texts = hf_split["text"]
    labels = np.array(hf_split["label"], dtype=np.int32)
    target_labels = np.arange(num_classes, dtype=np.int32)

    ds = TransformersDataset.from_arrays(
        texts,
        labels,
        tokenizer,
        target_labels=target_labels,
        max_length=max_length,
    )
    return ds



def make_classifier_factory(num_classes: int, cfg: ALConfig):
    model_args = TransformerModelArguments(c.TRANSFORMER_MODEL_NAME)

    clf_factory = TransformerBasedClassificationFactory(
        model_args,
        num_classes,
        kwargs=dict(
            device=c.DEVICE,
            num_epochs=cfg.num_epochs,
            mini_batch_size=cfg.train_batch_size,
        ),
    )
    return clf_factory


def make_query_strategy(name: QueryStrategy, num_classes: int):
    if name == QueryStrategy.RANDOM:
        return RandomSampling()
    if name == QueryStrategy.LEAST_CONFIDENCE:
        return LeastConfidence()
    if name == QueryStrategy.BALD:
        return BALD(dropout_samples=10)
    if name == QueryStrategy.BADGE:
        return BADGE(num_classes=num_classes)

def subsample_pool(hf_train_split: Dataset, pool_size: int, seed: int):
    """Сэмплируем pool_size из train."""
    n = len(hf_train_split)
    pool_size = min(pool_size, n)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    selected = indices[:pool_size]
    return hf_train_split.select(selected.tolist()), selected


def get_eval_split(ds_std):
    if "validation" in ds_std:
        return ds_std["validation"]
    else:
        return ds_std["test"]
    


def evaluate_on_test(active_learner, test_ds):
    y_pred = active_learner.classifier.predict(test_ds)
    y_true = test_ds.y

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return dict(acc=acc, macro_f1=macro_f1)

def mangle_experiment_name(
        dataset_key: str, 
        strategy_name: QueryStrategy, 
        seed: int, 
        oracle_type: OracleType | None = None
) -> str :
    if oracle_type is None: 
        return f"{dataset_key}_{strategy_name}_{seed}"
    return f"{dataset_key}_{strategy_name}_{seed}_{oracle_type}"

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

def run_all_strategies_for_dataset(
    tokenizer,
    dataset_key: str,
    standardized_datasets: Dict[str, Any],
    cfg: ALConfig,
    oracle_strategy: OracleStrategy | None = None
):
    """
    Запускает random / least_conf / bald / badge для одного датасета.
    prefix — префикс для имени эксперимента в логах (например, 'ag_news').
    """
    histories = {}

    for strategy_name in QueryStrategy:
        history = run_al_experiment_one(
            tokenizer,
            dataset_key=dataset_key,
            standardized_datasets=standardized_datasets,
            strategy_name=strategy_name,
            cfg=cfg,
            oracle_strategy=oracle_strategy,
        )

        histories[strategy_name] = history

    return histories


def run_all_seeds_and_strategies_for_dataset(
    tokenizer,
    dataset_key: str,
    standardized_datasets: Dict[str, Any],
    cfg: ALConfig,
    seeds: list[int], 
):
    """
    Запускает random / least_conf / bald / badge для одного датасета.
    prefix — префикс для имени эксперимента в логах (например, 'ag_news').
    """
    historiess = {}

    for seed in seeds:
        histories = run_all_strategies_for_dataset(
            tokenizer,
            dataset_key=dataset_key,
            standardized_datasets=standardized_datasets,
            cfg=dataclasses.replace(cfg, seed=seed),
        )

        historiess.update({(strategy, seed): history for strategy, history in histories.items()})
    return historiess

def plot_single_history(history: pd.DataFrame, title_prefix: str):
    df = history
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
    for axe in axes.flat: axe.grid(True)
    # 1. Accuracy vs число размеченных объектов
    axes[0,0].plot(df["labeled"], df["acc"], 'o--', ms=4, lw=0.8)  
    axes[0,0].set(xlabel = "Число размеченных объектов", ylabel = "Accuracy (validation)")
    axes[0,0].set_title(f"{title_prefix}: Accuracy vs #labeled")

    # 2. Accuracy vs стоимость
    axes[0,1].plot(df["cost_human"], df["acc"], 'o--', ms=4, lw=0.8)
    axes[0,1].set(xlabel = "Стоимость (человеческая разметка)", ylabel = "Accuracy (validation)")
    axes[0,1].set_title(f"{title_prefix}: Accuracy vs cost")

    # 3. F1 vs число размеченных объектов
    axes[1,0].plot(df["labeled"], df["macro_f1"], 'o--', ms=4, lw=0.8)  
    axes[1,0].set(xlabel = "Число размеченных объектов", ylabel = "F1 (validation)")
    axes[1,0].set_title(f"{title_prefix}: F1 vs #labeled")

    # 4. F1 vs стоимость
    axes[1,1].plot(df["cost_human"], df["macro_f1"], 'o--', ms=4, lw=0.8)
    axes[1,1].set(xlabel = "Стоимость (человеческая разметка)", ylabel = "F1 (validation)")
    axes[1,1].set_title(f"{title_prefix}: F1 vs cost")
    plt.show()


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



def llm_label_single(text: str, dataset_key: str, client: GigaChat) -> int:
    info = c.LABEL_SPACES[dataset_key]
    label2id = info["label2id"]
    labels = list(label2id.keys())
    labels_lower = {k.lower(): v for k, v in label2id.items()}
    task_desc = info["task_description"]

    prompt = (
        "Ты выступаешь как аккуратный разметчик датасета по классификации текста.\n"
        "Твоя задача — отнести текст ровно к одному из классов.\n"
        "Отвечай строго в формате JSON: {\"label\": \"<имя_класса>\"} без лишнего текста.\n"
        "Никогда не придумывай новые классы. Если текст кажется нейтральным или неоднозначным,\n"
        "всё равно выбери один из перечисленных классов, который ближе всего по смыслу.\n\n"
        f"{task_desc}\n\n"
        f"Возможные классы: {', '.join(labels)}.\n\n"
        f"Текст:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Выбери наиболее подходящий класс и верни только JSON. Кроме JSON ни в коем случем ничего больше не дописывай в ответе!!!"
    )

    response = client.chat(prompt)
    msg = response.choices[0].message
    
    content = msg.content.strip()

    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

    json_str = None
    start = content.find("{")
    end = content.find("}", start + 1) if start != -1 else -1
    if start != -1 and end != -1:
        json_str = content[start:end + 1]
    else:
        json_str = content

    try:
        data = json.loads(json_str)
        label_name = str(data.get("label", "")).strip()
    except Exception:
        m = re.search(r'"label"\s*:\s*"([^"]+)"', content)
        if m:
            label_name = m.group(1).strip()
        else:
            label_name = content.strip()

    label_key = label_name.lower()

    if label_key in labels_lower:
        return labels_lower[label_key]

    if dataset_key == "sst2":
        if "neutral" in label_key:
            if "negative" in labels_lower:
                return labels_lower["negative"]

        if "pos" in label_key and "positive" in labels_lower:
            return labels_lower["positive"]
        if "neg" in label_key and "negative" in labels_lower:
            return labels_lower["negative"]

    print(f"[WARN] Unknown LLM label '{label_name}', falling back to '{labels[0]}'")
    first_label_id = label2id[labels[0]]
    return first_label_id

def llm_label_batch(texts, dataset_key: str, client: GigaChat) -> list[int]:
    labels = []
    for t in texts:
        labels.append(llm_label_single(t, dataset_key, client))
    return labels


def get_oracle_labels(
    queried_indices,
    pool_ds,
    pool_hf,
    dataset_key: str,
    current_k: int,
    oracle_strategy: OracleStrategy,
):
    """
    Возвращает метки для индексов queried_indices в зависимости от типа оракула.

    oracle_type:
      - "human"  -> берём gold labels из pool_ds
      - "llm"    -> спрашиваем LLM (GigaChat) через llm_label_batch(...)
    """
    oracle_type = oracle_strategy.to_real_oracle_type(current_k)

    if oracle_type == OracleType.HUMAN:
        # идеальный человек: просто gold labels
        return pool_ds.y[queried_indices]

    elif oracle_type == OracleType.LLM:
        gigachat_client = oracle_strategy.gigachat_client
        # pool_hf — это HF Dataset с полем "text"
        texts = [pool_hf[int(i)]["text"] for i in queried_indices]
        labels_llm = llm_label_batch(texts, dataset_key, gigachat_client)
        return np.array(labels_llm, dtype=np.int64)

    else:
        raise ValueError(f"Unknown oracle_type={oracle_type}")


def run_al_experiment_one_oracles(
    dataset_key: str,
    standardized_datasets: Dict[str, Any],
    strategy_name: str,
    cfg: ALConfig,
    experiment_name: Optional[str] = None,
    save_checkpoints: bool = False,
    save_history_flag: bool = True,
    oracle_type: str = "human",      # 'human' | 'llm' | 'hybrid'
    hybrid_k: Optional[int] = None,  # сколько итераций считать "LLM-фазой" при oracle_type='hybrid'
    gigachat_client=None,            # экземпляр клиента GigaChat для LLM-оракула
):
    """
    dataset_key: 'sst2' / 'ag_news' / 'hatexplain'
    strategy_name: 'random', 'least_conf', 'bald', 'badge' (если используешь)
    oracle_type:
        - 'human'  -> gold labels
        - 'llm'    -> все метки даёт LLM
        - 'hybrid' -> первые hybrid_k итераций LLM, дальше human
    """
    if experiment_name is None:
        experiment_name = f"{dataset_key}_{strategy_name}_{oracle_type}_{cfg.seed}"

    print(
        f"\n=== AL EXPERIMENT: dataset={dataset_key}, "
        f"strategy={strategy_name}, oracle={oracle_type}, exp={experiment_name} ==="
    )

    ds_std = standardized_datasets[dataset_key]
    train_hf = ds_std["train"]
    eval_hf = get_eval_split(ds_std)

    num_classes = len(set(train_hf["label"]))

    pool_hf, _ = subsample_pool(train_hf, cfg.pool_size, cfg.seed)

    pool_ds = to_transformers_dataset(pool_hf, num_classes, max_length=cfg.max_length)
    test_ds = to_transformers_dataset(eval_hf, num_classes, max_length=cfg.max_length)

    clf_factory = make_classifier_factory(num_classes, cfg)
    query_strategy = make_query_strategy(strategy_name, num_classes)

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, pool_ds)

    indices_initial = random_initialization_balanced(
        pool_ds.y, n_samples=cfg.initial_labeled
    )
    y_initial = np.array([pool_ds.y[i] for i in indices_initial])
    active_learner.initialize_data(indices_initial, y_initial)

    history: List[Dict[str, Any]] = []
    indices_labeled = indices_initial.copy()

    for it in range(cfg.iterations + 1):
        # 1) оценка
        metrics = evaluate_on_test(active_learner, test_ds)
        labeled_count = len(indices_labeled)

        # считаем стоимость разметки (пока без раздельного учёта гибрида — это можно позже доработать)
        total_cost_human = labeled_count * cfg.cost_human
        total_cost_llm = labeled_count * cfg.cost_llm

        step_info = dict(
            iter=int(it),
            labeled=int(labeled_count),
            acc=float(metrics["acc"]),
            macro_f1=float(metrics["macro_f1"]),
            cost_human=float(total_cost_human),
            cost_llm=float(total_cost_llm),
        )
        history.append(step_info)

        print(
            f"[Iter {it:02d}] labeled={labeled_count:4d} | "
            f"acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
            f"cost_human={total_cost_human:.1f}"
        )

        # сохраняем чекпоинт модели после оценки
        if save_checkpoints:
            save_active_learner_checkpoint(active_learner, experiment_name, iteration=it)

        # последняя "итерация-оценка" — дальше не запрашиваем новые примеры
        if it == cfg.iterations:
            break

        # 2) запрос новой порции примеров
        queried_indices = active_learner.query(num_samples=cfg.batch_size)

        # 3) выбираем, какого оракула использовать на этой итерации
        if oracle_type == "hybrid":
            if hybrid_k is None:
                raise ValueError("oracle_type='hybrid', но hybrid_k не задан")
            current_oracle = "llm" if it < hybrid_k else "human"
        else:
            current_oracle = oracle_type

        # 4) получаем метки от текущего оракула
        y_queried = get_oracle_labels(
            queried_indices,
            pool_ds=pool_ds,
            pool_hf=pool_hf,
            dataset_key=dataset_key,
            oracle_type=current_oracle,
            gigachat_client=gigachat_client,
        )

        active_learner.update(y_queried)
        indices_labeled = np.concatenate([indices_labeled, queried_indices])

    if save_history_flag:
        save_history(history, experiment_name)

    return history


def encode_texts_to_embeddings(texts, batch_size: int = 32, max_length: int = 128):
    """
    Кодирует список текстов в CLS-эмбеддинги DistilBERT.
    Возвращает np.array формы [N, hidden_size].
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = c.EMB_TOKENIZER(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(c.DEVICE)

        with torch.no_grad():
            outputs = c.EMB_MODEL(**enc)
            # CLS-токен у DistilBERT — это первый токен
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
            all_embs.append(cls_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)

def select_initial_indices_diversity(pool_hf, n_samples: int, seed: int = 42, max_length: int = 128):
    """
    Выбирает n_samples индексов из HF Dataset (pool_hf) на основе k-means по CLS-эмбеддингам.
    """
    texts = [pool_hf[i]["text"] for i in range(len(pool_hf))]
    emb = encode_texts_to_embeddings(texts, batch_size=32, max_length=max_length)

    kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init=10)
    kmeans.fit(emb)

    # Для каждого кластера находим точку, ближайшую к центроиду
    indices = []
    for k in range(n_samples):
        cluster_mask = (kmeans.labels_ == k)
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue  # на всякий случай
        cluster_embs = emb[cluster_indices]
        center = kmeans.cluster_centers_[k]
        dists = np.linalg.norm(cluster_embs - center, axis=1)
        best_idx = cluster_indices[np.argmin(dists)]
        indices.append(int(best_idx))

    return np.array(indices, dtype=np.int64)


def run_al_experiment_diversity_init(
    strategy_name: str,
    cfg: ALConfig,
    experiment_name: str,
    oracle_type: str = "human",          # 'human' | 'llm' | 'hybrid'
    dataset_key: str = "sst2",
    hybrid_k: Optional[int] = None,
    gigachat_client=None,
    initial_oracle_type: str = "human",  # 'human' | 'llm'
):
    """
    AL-эксперимент с diversity-based инициализацией (KMeans по эмбеддингам),
    с поддержкой разных типов оракулов и честным учётом стоимости.

    strategy_name: 'random' / 'least_conf' / 'bald' / 'badge' (если очень захочешь)
    oracle_type:
      - 'human'  -> все последующие батчи размечает человек
      - 'llm'    -> все батчи размечает LLM
      - 'hybrid' -> первые hybrid_k итераций LLM, дальше человек
    initial_oracle_type:
      - 'human'  -> стартовые 200 меток считаем от человека
      - 'llm'    -> стартовые 200 меток считаем от LLM
    """
    print(
        f"\n=== AL EXPERIMENT (DIVERSITY INIT): dataset={dataset_key}, "
        f"strategy={strategy_name}, oracle={oracle_type}, "
        f"init_oracle={initial_oracle_type}, exp={experiment_name} ==="
    )

    ds_std = standardized_datasets[dataset_key]
    train_hf = ds_std["train"]
    eval_hf, eval_name = get_eval_split(ds_std)

    num_classes = len(set(train_hf["label"]))
    print(f"Num classes: {num_classes}, eval split: {eval_name}")

    pool_hf, pool_indices = subsample_pool(train_hf, cfg.pool_size, cfg.seed)
    print(f"Pool size (subsampled): {len(pool_hf)} (original train: {len(train_hf)})")

    pool_ds = to_transformers_dataset(pool_hf, num_classes, max_length=cfg.max_length)
    test_ds = to_transformers_dataset(eval_hf, num_classes, max_length=cfg.max_length)

    clf_factory = make_classifier_factory(num_classes, cfg)
    query_strategy = make_query_strategy(strategy_name, num_classes)
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, pool_ds)

    # --- DIVERSITY-BASED INITIALIZATION ---
    indices_initial = select_initial_indices_diversity(
        pool_hf,
        n_samples=cfg.initial_labeled,
        seed=cfg.seed,
        max_length=cfg.max_length,
    )

    # метки для initial набора
    if initial_oracle_type == "human":
        y_initial = np.array([pool_ds.y[i] for i in indices_initial])
    elif initial_oracle_type == "llm":
        if gigachat_client is None:
            raise ValueError("initial_oracle_type='llm', но gigachat_client=None")
        texts_init = [pool_hf[int(i)]["text"] for i in indices_initial]
        y_initial = np.array(
            llm_label_batch(texts_init, dataset_key, gigachat_client),
            dtype=np.int64,
        )
    else:
        raise ValueError(f"Unknown initial_oracle_type={initial_oracle_type}")

    active_learner.initialize_data(indices_initial, y_initial)
    print(f"Initial labeled (diversity): {len(indices_initial)}")

    # счётчики стоимости
    cum_human_labels = 0
    cum_llm_labels = 0

    if initial_oracle_type == "human":
        cum_human_labels += len(indices_initial)
    else:
        cum_llm_labels += len(indices_initial)

    history: List[Dict[str, Any]] = []
    indices_labeled = indices_initial.copy()

    for it in range(cfg.iterations + 1):
        # 1) оценка
        metrics = evaluate_on_test(active_learner, test_ds)
        labeled_count = len(indices_labeled)

        total_cost = (
            cum_human_labels * cfg.cost_human
            + cum_llm_labels * cfg.cost_llm
        )

        step_info = dict(
            iter=int(it),
            labeled=int(labeled_count),
            acc=float(metrics["acc"]),
            macro_f1=float(metrics["macro_f1"]),
            cum_human_labels=int(cum_human_labels),
            cum_llm_labels=int(cum_llm_labels),
            total_cost=float(total_cost),
        )
        history.append(step_info)

        print(
            f"[Iter {it:02d}] labeled={labeled_count:4d} | "
            f"acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
            f"human_labels={cum_human_labels} | llm_labels={cum_llm_labels} | "
            f"total_cost={total_cost:.1f}"
        )

        if it == cfg.iterations:
            break

        # 2) запрос новой порции примеров
        queried_indices = active_learner.query(num_samples=cfg.batch_size)

        # 3) выбираем, какого оракула использовать на этой итерации
        if oracle_type == "hybrid":
            if hybrid_k is None:
                raise ValueError("oracle_type='hybrid', но hybrid_k не задан")
            current_oracle = "llm" if it < hybrid_k else "human"
        else:
            current_oracle = oracle_type

        # 4) получаем метки от текущего оракула
        y_queried = get_oracle_labels(
            queried_indices,
            pool_ds=pool_ds,
            pool_hf=pool_hf,
            dataset_key=dataset_key,
            oracle_type=current_oracle,
            gigachat_client=gigachat_client,
        )

        # 5) обновляем счётчики стоимости
        if current_oracle == "human":
            cum_human_labels += len(queried_indices)
        elif current_oracle == "llm":
            cum_llm_labels += len(queried_indices)
        else:
            raise ValueError(f"Unexpected current_oracle={current_oracle}")

        # 6) обновляем AL-модель
        active_learner.update(y_queried)
        indices_labeled = np.concatenate([indices_labeled, queried_indices])

    save_history(history, experiment_name)
    return history



def load_history_pd(exp_name: str) -> pd.DataFrame:
    """Читает CSV вида <exp_name>_history.csv из LOG_DIR.
       Если файла нет, возвращает None.
    """
    path = os.path.join(c.LOG_DIR, f"{exp_name}_history.csv")
    if not os.path.exists(path):
        print(f"[WARN] history csv not found: {path}")
        return None
    df = pd.read_csv(path)
    return df

def load_history_dict(exp_name: str) -> pd.DataFrame:
    df = load_history_pd(exp_name)
    return df.to_dict('records')

def get_cost_column(df: pd.DataFrame) -> str:
    """Выбираем, по какому столбцу рисовать 'стоимость':
       сначала total_cost (для LLM/гибридов), иначе cost_human.
    """
    if "total_cost" in df.columns:
        return "total_cost"
    elif "cost_human" in df.columns:
        return "cost_human"
    else:
        raise ValueError("No suitable cost column found in history dataframe")

def plot_oracle_comparison_sst2(
    strategy_name: str,
    human_exp: str | None,
    llm_exp: str | None,
    hybrid_exp: str | None,
    title_prefix: str
):
    """
    Рисует два графика для одной стратегии:
      1) accuracy vs #labeled
      2) accuracy vs cost
    для разных типов оракулов (human / llm / hybrid), если соответствующие
    эксперименты существуют.
    """
    experiments = {
        "human": human_exp,
        "llm-only": llm_exp,
        "hybrid (k=3)": hybrid_exp,
    }

    dfs = {}
    for label, exp_name in experiments.items():
        if exp_name is None:
            continue
        df = load_history_pd(exp_name)
        if df is not None:
            dfs[label] = df

    if not dfs:
        print(f"[WARN] no histories found for {strategy_name}")
        return

    # --- 1. Accuracy vs #labeled ---
    plt.figure(figsize=(6,4))
    for label, df in dfs.items():
        plt.plot(df["labeled"], df["acc"], marker="o", label=label)
    plt.xlabel("Число размеченных объектов")
    plt.ylabel("Accuracy (validation)")
    plt.title(f"{title_prefix}: accuracy vs #labeled")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2. Accuracy vs cost ---
    plt.figure(figsize=(6,4))
    for label, df in dfs.items():
        cost_col = get_cost_column(df)
        plt.plot(df[cost_col], df["acc"], marker="o", label=label)
    plt.xlabel("Суммарная стоимость разметки")
    plt.ylabel("Accuracy (validation)")
    plt.title(f"{title_prefix}: accuracy vs cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# def plot_diversity_init_comparison_sst2():
#     experiments = {
#         "baseline (random, human)": "sst2_random",
#         "diversity-init, human": "sst2_random_diversityinit_human",
#         "diversity-init, LLM-only": "sst2_random_diversityinit_llm",
#         "diversity-init, hybrid (k=3)": "sst2_random_diversityinit_hybrid_k3",
#     }

#     dfs = {}
#     for label, exp_name in experiments.items():
#         df = load_history_pd(exp_name)
#         if df is not None:
#             dfs[label] = df

#     # --- Accuracy vs #labeled ---
#     plt.figure(figsize=(6,4))
#     for label, df in dfs.items():
#         plt.plot(df["labeled"], df["acc"], marker="o", label=label)
#     plt.xlabel("Число размеченных объектов")
#     plt.ylabel("Accuracy (validation)")
#     plt.title("SST-2, Random: обычный init vs diversity-init")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # --- Accuracy vs cost (total_cost / cost_human) ---
#     plt.figure(figsize=(6,4))
#     for label, df in dfs.items():
#         cost_col = get_cost_column(df)
#         plt.plot(df[cost_col], df["acc"], marker="o", label=label)
#     plt.xlabel("Суммарная стоимость разметки")
#     plt.ylabel("Accuracy (validation)")
#     plt.title("SST-2, Random: стоимость при разных init и оракулах")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()




# def encode_texts_to_embeddings(texts: List[str], batch_size: int = c.DIV_CFG.emb_batch_size,
#                                max_length: int = c.DIV_CFG.emb_max_length) -> np.ndarray:
#     """
#     Кодирует список текстов в CLS-эмбеддинги.
#     Возвращает np.array формы [N, hidden_size].
#     """
#     if emb_model is None:
#         raise ValueError("Embedding model is not loaded")

#     all_embs = []
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         enc = EMB_TOKENIZER(
#             batch_texts,
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(c.DEVICE)

#         with torch.no_grad():
#             outputs = emb_model(**enc)
#             # CLS-токен - первый токен
#             cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
#             all_embs.append(cls_emb.cpu().numpy())

#     return np.concatenate(all_embs, axis=0)

# def select_initial_indices_diversity(pool_hf, n_samples: int = c.DIV_CFG.diversity_samples,
#                                      seed: int = 42) -> np.ndarray:
#     """
#     Выбирает n_samples индексов из HF Dataset на основе k-means по CLS-эмбеддингам.
#     """

#     texts = [pool_hf[i]["text"] for i in range(len(pool_hf))]
#     print(f"Encoding {len(texts)} texts to embeddings...")

#     try:
#         emb = encode_texts_to_embeddings(texts, max_length=c.DIV_CFG.emb_max_length)

#         kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init=10)
#         kmeans.fit(emb)

#         # Для каждого кластера находим точку, ближайшую к центроиду
#         indices = []
#         for k in range(n_samples):
#             cluster_mask = (kmeans.labels_ == k)
#             cluster_indices = np.where(cluster_mask)[0]
#             if len(cluster_indices) == 0:
#                 continue
#             cluster_embs = emb[cluster_indices]
#             center = kmeans.cluster_centers_[k]
#             dists = np.linalg.norm(cluster_embs - center, axis=1)
#             best_idx = cluster_indices[np.argmin(dists)]
#             indices.append(int(best_idx))

#         print(f"Selected {len(indices)} initial indices by diversity.")
#         return np.array(indices, dtype=np.int64)

#     except Exception as e:
#         print(f" Diversity initialization failed: {e}")
#         print("Falling back to random balanced initialization...")
#         # Возвращаем случайные сбалансированные индексы
#         return random_initialization_balanced(
#             np.array([pool_hf[i]["label"] for i in range(len(pool_hf))]),
#             n_samples=n_samples
#         )
    
def get_prompt_for_dataset(dataset_key: str, text: str) -> str:
    """
    Возвращает промпт для конкретного датасета.
    """
    if dataset_key == "sst2":
        return f"""Текст: "{text}"

Этот текст выражает позитивное или негативное мнение?
Выбери один из вариантов:
1. негативное (метка 0)
2. позитивное (метка 1)

Ответь только числом (0 или 1)."""

    elif dataset_key == "ag_news":
        return f"""Текст: "{text}"

К какой категории относится этот текст новостей?
Выбери один из вариантов:
1. Мировые новости (World) (метка 0)
2. Спорт (Sports) (метка 1)
3. Бизнес (Business) (метка 2)
4. Наука и технологии (Sci/Tech) (метка 3)

Ответь только числом (0, 1, 2 или 3)."""

    else:
        return f"""Классифицируй текст: "{text}"

Верни только числовую метку (0, 1, 2, ...)."""

def extract_label_from_response(response: str) -> Optional[int]:
    """
    Извлекает числовую метку из ответа LLM.
    """
    import re

    numbers = re.findall(r'\d+', response)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            pass

    if "0" in response or "негатив" in response.lower() or "negative" in response.lower():
        return 0
    elif "1" in response or "позитив" in response.lower() or "positive" in response.lower():
        return 1
    elif "2" in response:
        return 2
    elif "3" in response:
        return 3

    return None

# def llm_label_single(text: str, dataset_key: str, max_retries: int = 3) -> Optional[int]:
#     """
#     Размечает один текст с помощью LLM.
#     """
#     cache_key = f"{dataset_key}_{hash(text)}"
#     cache_file = os.path.join(DIV_CFG.llm_cache_dir, f"{cache_key}.txt")

#     if DIV_CFG.use_cached_llm_labels and os.path.exists(cache_file):
#         try:
#             with open(cache_file, 'r', encoding='utf-8') as f:
#                 cached_label = int(f.read().strip())
#             print(f"  ↳ Использована метка из кэша: {cached_label}")
#             return cached_label
#         except:
#             pass

#     prompt = get_prompt_for_dataset(dataset_key, text)

#     for attempt in range(max_retries):
#         try:
#             time.sleep(0.5)

#             response = client.chat(prompt)
#             label = extract_label_from_response(response.choices[0].message.content)

#             if label is not None:
#                 with open(cache_file, 'w', encoding='utf-8') as f:
#                     f.write(str(label))
#                 return label
#             else:
#                 print(f"Не удалось извлечь метку из ответа: {response.choices[0].message.content[:100]}...")

#         except Exception as e:
#             print(f"Ошибка при запросе к LLM (попытка {attempt + 1}): {e}")
#             time.sleep(1)  # Ждем перед повторной попыткой

#     print(f"Не удалось получить метку для текста после {max_retries} попыток")
#     return None

# def llm_label_batch(texts: List[str], dataset_key: str, batch_size: int = 10) -> List[Optional[int]]:
#     """
#     Размечает батч текстов с помощью LLM.
#     """
#     labels = []
#     for i, text in enumerate(texts):
#         print(f"  LLM labeling {i+1}/{len(texts)}...")
#         label = llm_label_single(text, dataset_key)
#         labels.append(label)

#         # Небольшая задержка между запросами
#         if i < len(texts) - 1:
#             time.sleep(0.2)

#     return labels

# def get_oracle_labels(indices: np.ndarray, pool_hf, dataset_key: str,
#                       oracle_type: str = "human", client=None) -> np.ndarray:
#     """
#     Получает метки от указанного оракула.

#     Args:
#         indices: Индексы в пуле для разметки
#         pool_hf: Датасет пула
#         dataset_key: Ключ датасета
#         oracle_type: Тип оракула ("human", "llm", "hybrid")
#         client: Клиент LLM (только для llm/hybrid)

#     Returns:
#         Массив меток
#     """
#     texts = [pool_hf[int(i)]["text"] for i in indices]

#     if oracle_type == "human":
#         # Используем истинные метки из датасета (эмулируем человеческую разметку)
#         return np.array([pool_hf[int(i)]["label"] for i in indices], dtype=np.int64)

#     elif oracle_type == "llm":
#         if not GIGACHAT_AVAILABLE:
#             print(" GigaChat не доступен, используем человеческие метки")
#             return np.array([pool_hf[int(i)]["label"] for i in indices], dtype=np.int64)

#         print(f"  Запрашиваем {len(texts)} меток у LLM...")
#         labels = llm_label_batch(texts, dataset_key)

#         # Заменяем None на человеческие метки
#         final_labels = []
#         for i, label in enumerate(labels):
#             if label is None:
#                 print(f"  LLM не вернул метку для примера {i}, используем человеческую")
#                 final_labels.append(pool_hf[int(indices[i])]["label"])
#             else:
#                 final_labels.append(label)

#         return np.array(final_labels, dtype=np.int64)

#     else:
#         raise ValueError(f"Неизвестный тип оракула: {oracle_type}")

# def run_single_diversity_experiment(
#     dataset_key: str,
#     strategy_name: str,
#     seed: int,
#     cfg: ALConfig,
#     div_cfg: DiversityConfig = c.DIV_CFG,
#     oracle_type: str = "hybrid",
#     save_history: bool = True,
#     verbose: bool = True,
# ):
#     """
#     Запуск одного эксперимента с diversity инициализацией и LLM разметкой.
#     """

#     start_time = time.time()
#     experiment_name = f"{dataset_key}_{strategy_name}_div_{oracle_type}_seed{seed}"

#     print(f"\n STARTING DIVERSITY EXPERIMENT: {experiment_name}")
#     print(f"  Oracle type: {oracle_type}")
#     print(f"  Diversity init: {div_cfg.use_diversity_init}")

#     # Установка сида
#     set_seed(seed)

#     # Загрузка данных
#     ds_std = standardized_datasets[dataset_key]
#     train_hf = ds_std["train"]
#     eval_hf, eval_name = get_eval_split(ds_std)

#     num_classes = len(set(train_hf["label"]))

#     # Сэмплирование пула
#     pool_hf, pool_indices = subsample_pool(train_hf, cfg.pool_size, seed)
#     print(f"  Pool size: {len(pool_hf)}")

#     # Подготовка датасетов
#     pool_ds = to_transformers_dataset(pool_hf, num_classes, max_length=cfg.max_length)
#     test_ds = to_transformers_dataset(eval_hf, num_classes, max_length=cfg.max_length)

#     # Создание классификатора и стратегии
#     clf_factory = make_classifier_factory(num_classes, cfg)
#     query_strategy = make_query_strategy(strategy_name, num_classes)

#     # Инициализация Active Learner
#     active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, pool_ds)

#     # Diversity инициализация
#     if div_cfg.use_diversity_init:
#         indices_initial = select_initial_indices_diversity(
#             pool_hf,
#             n_samples=cfg.initial_labeled,
#             seed=seed
#         )
#     else:
#         indices_initial = random_initialization_balanced(
#             pool_ds.y,
#             n_samples=cfg.initial_labeled
#         )

#     # Получение начальных меток от оракула
#     if oracle_type == "human":
#         y_initial = np.array([pool_ds.y[i] for i in indices_initial])
#     elif oracle_type == "llm" or oracle_type == "hybrid":
#         if not GIGACHAT_AVAILABLE:
#             print("  GigaChat не доступен, используем человеческие метки")
#             y_initial = np.array([pool_ds.y[i] for i in indices_initial])
#         else:
#             print("  Получаем начальные метки от LLM...")
#             texts_initial = [pool_hf[int(i)]["text"] for i in indices_initial]
#             y_initial = np.array(
#                 llm_label_batch(texts_initial, dataset_key),
#                 dtype=np.int64
#             )

#             # Проверяем качество LLM разметки
#             true_labels = np.array([pool_ds.y[i] for i in indices_initial])
#             if len(y_initial) == len(true_labels):
#                 accuracy = np.mean(y_initial == true_labels)
#                 print(f"  Качество LLM разметки на начальном наборе: {accuracy:.2%}")
#     else:
#         y_initial = np.array([pool_ds.y[i] for i in indices_initial])

#     active_learner.initialize_data(indices_initial, y_initial)

#     history = []
#     indices_labeled = indices_initial.copy()

#     # Счетчики стоимости
#     cum_human_labels = 0
#     cum_llm_labels = 0

#     # Определяем, какие метки считать человеческими, а какие LLM
#     if oracle_type == "human":
#         cum_human_labels += len(indices_initial)
#     elif oracle_type == "llm":
#         cum_llm_labels += len(indices_initial)
#     elif oracle_type == "hybrid":
#         # В гибридном режиме начальные метки считаем человеческими
#         cum_human_labels += len(indices_initial)

#     # AL итерации
#     for it in range(cfg.iterations + 1):
#         # Оценка модели
#         metrics = evaluate_on_test(active_learner, test_ds)
#         labeled_count = len(indices_labeled)

#         total_cost = (
#             cum_human_labels * cfg.cost_human
#             + cum_llm_labels * cfg.cost_llm
#         )

#         step_info = {
#             'iter': it,
#             'seed': seed,
#             'labeled': labeled_count,
#             'acc': metrics['acc'],
#             'macro_f1': metrics['macro_f1'],
#             'cum_human_labels': cum_human_labels,
#             'cum_llm_labels': cum_llm_labels,
#             'total_cost': total_cost,
#             'oracle_type': oracle_type,
#             'diversity_init': div_cfg.use_diversity_init
#         }
#         history.append(step_info)

#         if verbose and it % 5 == 0:
#             print(f"  [Iter {it:02d}] labeled={labeled_count:4d} | "
#                   f"acc={metrics['acc']:.4f} | cost={total_cost:.1f} | "
#                   f"human={cum_human_labels} | llm={cum_llm_labels}")

#         if it == cfg.iterations:
#             break

#         # Запрос новых примеров
#         queried_indices = active_learner.query(num_samples=cfg.batch_size)

#         # Определяем, какой оракул использовать на этой итерации
#         current_oracle = oracle_type
#         if oracle_type == "hybrid":
#             if it < div_cfg.hybrid_k:
#                 current_oracle = "llm"
#             else:
#                 current_oracle = "human"

#         # Получаем метки от оракула
#         y_queried = get_oracle_labels(
#             queried_indices,
#             pool_hf=pool_hf,
#             dataset_key=dataset_key,
#             oracle_type=current_oracle,
#             client=client
#         )

#         # Обновляем счетчики стоимости
#         if current_oracle == "human":
#             cum_human_labels += len(queried_indices)
#         elif current_oracle == "llm":
#             cum_llm_labels += len(queried_indices)

#         # Обновляем модель
#         active_learner.update(y_queried)
#         indices_labeled = np.concatenate([indices_labeled, queried_indices])

#     total_time = time.time() - start_time
#     print(f"Experiment completed in {total_time:.1f}s")

#     if save_history:
#         save_single_history(history, experiment_name)

#     return history

# def run_multiple_diversity_experiments(
#     dataset_key: str,
#     strategy_name: str,
#     seeds: List[int],
#     cfg: ALConfig = None,
#     div_cfg: DiversityConfig = c.DIV_CFG,
#     oracle_type: str = "hybrid",
#     verbose: bool = True,
# ):
#     """
#     Запуск нескольких diversity экспериментов с разными сидами.
#     """
#     if cfg is None:
#         cfg = ALConfig()

#     all_histories = []
#     failed_runs = []

#     print(f"\n{'='*80}")
#     print(f"RUNNING DIVERSITY EXPERIMENTS: {dataset_key.upper()} - {strategy_name.upper()}")
#     print(f"Oracle type: {oracle_type}")
#     print(f"Diversity init: {div_cfg.use_diversity_init}")
#     print('='*80)

#     for i, seed in enumerate(seeds, 1):
#         print(f"\n Diversity run {i}/{len(seeds)} (Seed: {seed})")

#         try:
#             history = run_single_diversity_experiment(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy_name,
#                 seed=seed,
#                 cfg=cfg,
#                 div_cfg=div_cfg,
#                 oracle_type=oracle_type,
#                 save_history=True,
#                 verbose=verbose
#             )
#             all_histories.append(history)
#             print(f"Success - Final accuracy: {history[-1]['acc']:.4f}")

#         except Exception as e:
#             print(f"   Failed - Error: {str(e)}")
#             failed_runs.append((seed, str(e)))

#     # Агрегация результатов
#     if all_histories:
#         aggregated = aggregate_histories(all_histories)

#         # Добавляем информацию о конфигурации
#         exp_name = f"{dataset_key}_{strategy_name}_div_{oracle_type}_aggregated"

#         # Используем обновленную функцию save_aggregated_history
#         save_aggregated_history(aggregated, exp_name, cfg)

#         return aggregated, all_histories, failed_runs
#     else:
#         return None, [], failed_runs

# def compare_oracle_strategies(
#     dataset_key: str,
#     strategies: List[str],
#     seeds: List[int],
#     cfg: ALConfig = None,
#     div_cfg: DiversityConfig = c.DIV_CFG,
#     oracle_types: List[str] = ["human", "llm", "hybrid"],
#     use_cache: bool = True,
# ):
#     """
#     Сравнение стратегий с разными типами оракулов.
#     """
#     if cfg is None:
#         cfg = ALConfig()

#     all_results = {}

#     print(f"\n{'#'*80}")
#     print(f"COMPARING ORACLE STRATEGIES: {dataset_key.upper()}")
#     print(f"Strategies: {', '.join(strategies)}")
#     print(f"Oracle types: {', '.join(oracle_types)}")
#     print(f"Diversity init: {div_cfg.use_diversity_init}")
#     print('#'*80)

#     # Если GigaChat не доступен, убираем LLM и гибридные стратегии
#     if not GIGACHAT_AVAILABLE:
#         print(" GigaChat не доступен, используем только человеческий оракул")
#         oracle_types = ["human"]

#     for oracle_type in oracle_types:
#         print(f"\n{'='*80}")
#         print(f"ORACLE TYPE: {oracle_type.upper()}")
#         print('='*80)

#         oracle_results = {}

#         for strategy in strategies:
#             print(f"\nProcessing {strategy} with {oracle_type} oracle...")

#             # Проверяем кэш
#             exp_name = f"{dataset_key}_{strategy}_div_{oracle_type}_aggregated"
#             aggregated_path = os.path.join(AGGREGATED_LOG_DIR, f"{exp_name}.json")

#             if use_cache and os.path.exists(aggregated_path):
#                 print(f"   Loading from cache...")
#                 try:
#                     with open(aggregated_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)

#                     # Извлекаем результаты
#                     if isinstance(data, dict) and 'results' in data:
#                         aggregated = data['results']
#                     else:
#                         aggregated = data

#                     # Загружаем индивидуальные истории
#                     individual_histories = []
#                     for seed in seeds:
#                         individual_path = os.path.join(
#                             LOG_DIR,
#                             f"{dataset_key}_{strategy}_div_{oracle_type}_seed{seed}_history.csv"
#                         )
#                         if os.path.exists(individual_path):
#                             df = pd.read_csv(individual_path)
#                             history = df.to_dict('records')
#                             individual_histories.append(history)

#                     oracle_results[strategy] = {
#                         'aggregated': aggregated,
#                         'individual': individual_histories,
#                         'failed': [],
#                         'cached': True
#                     }

#                     print(f"Loaded from cache")
#                     continue

#                 except Exception as e:
#                     print(f"  Error loading from cache: {e}")

#             # Запускаем эксперимент
#             aggregated, individual, failed = run_multiple_diversity_experiments(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy,
#                 seeds=seeds,
#                 cfg=cfg,
#                 div_cfg=div_cfg,
#                 oracle_type=oracle_type,
#                 verbose=True
#             )

#             if aggregated:
#                 oracle_results[strategy] = {
#                     'aggregated': aggregated,
#                     'individual': individual,
#                     'failed': failed,
#                     'cached': False
#                 }

#         all_results[oracle_type] = oracle_results

#     return all_results

# def plot_oracle_comparison(dataset_key: str, all_results: Dict, strategy: str):
#     """
#     Визуализация сравнения разных типов оракулов для одной стратегии.
#     """
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(12, 8))

#     colors = {'human': 'blue', 'llm': 'red', 'hybrid': 'green'}

#     for oracle_type, oracle_results in all_results.items():
#         if strategy in oracle_results and oracle_results[strategy]['aggregated']:
#             df = pd.DataFrame(oracle_results[strategy]['aggregated'])
#             color = colors.get(oracle_type, 'black')

#             plt.plot(df['labeled'], df['acc_mean'],
#                      color=color, label=f'{oracle_type.upper()}', linewidth=2)

#             plt.fill_between(df['labeled'],
#                              df['acc_mean'] - df['acc_std'],
#                              df['acc_mean'] + df['acc_std'],
#                              alpha=0.2, color=color)

#     plt.xlabel('Number of Labeled Samples', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.title(f'{dataset_key.upper()} - {strategy.upper()}: Oracle Comparison\n(Mean ± Std)', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title='Oracle Type')
#     plt.tight_layout()

#     plot_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_{strategy}_oracle_comparison.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()

#     # График стоимости
#     plt.figure(figsize=(12, 8))

#     for oracle_type, oracle_results in all_results.items():
#         if strategy in oracle_results and oracle_results[strategy]['aggregated']:
#             df = pd.DataFrame(oracle_results[strategy]['aggregated'])
#             color = colors.get(oracle_type, 'black')

#             plt.plot(df['total_cost_mean'], df['acc_mean'],
#                      color=color, label=f'{oracle_type.upper()}', linewidth=2)

#     plt.xlabel('Total Cost', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.title(f'{dataset_key.upper()} - {strategy.upper()}: Cost-Effectiveness Comparison', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title='Oracle Type')
#     plt.tight_layout()

#     cost_plot_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_{strategy}_cost_comparison.png")
#     plt.savefig(cost_plot_path, dpi=300, bbox_inches='tight')
#     plt.show()

# def run_diversity_experiments_main():
#     """
#     Основная функция для запуска diversity экспериментов с GigaChat.
#     """
#     print("\n" + "="*80)
#     print("DIVERSITY ACTIVE LEARNING WITH GIGACHAT")
#     print("="*80)

#     if not GIGACHAT_AVAILABLE:
#         print(" WARNING: GigaChat is not available")
#         print("    Will run only human oracle experiments")

#     # Конфигурация для diversity экспериментов
#     cfg = ALConfig(
#         seed=42,
#         pool_size=POOL_SIZE,
#         initial_labeled=DIV_CFG.diversity_samples,  # Используем больше для diversity
#         batch_size=BATCH_SIZE,
#         iterations=N_ITER,
#         max_labeled=MAX_LABELED,
#         cost_human=COST_HUMAN,
#         cost_llm=COST_LLM,
#         max_length=MAX_LENGTH,
#         num_epochs=3,
#         train_batch_size=16,
#     )

#     # Стратегии для тестирования
#     strategies = ["random", "least_conf", "bald", "badge"]

#     # Типы оракулов
#     oracle_types = ["human", "llm", "hybrid"] if GIGACHAT_AVAILABLE else ["human"]

#     all_datasets_results = {}

#     # Запуск для каждого датасета
#     for dataset_key in DATASETS.keys():
#         print(f"\n{'#'*80}")
#         print(f"DIVERSITY EXPERIMENTS FOR: {dataset_key.upper()}")
#         print(f"GigaChat available: {GIGACHAT_AVAILABLE}")
#         print('#'*80)

#         results = compare_oracle_strategies(
#             dataset_key=dataset_key,
#             strategies=strategies,
#             seeds=SEEDS[:NUM_RUNS],
#             cfg=cfg,
#             div_cfg=DIV_CFG,
#             oracle_types=oracle_types,
#             use_cache=True
#         )

#         all_datasets_results[dataset_key] = results

#         # Визуализация для каждой стратегии
#         for strategy in strategies:
#             plot_oracle_comparison(dataset_key, results, strategy)

#         # Сводная статистика
#         save_oracle_summary(dataset_key, results)

#     print("\n" + "="*80)
#     print("DIVERSITY EXPERIMENTS COMPLETED")
#     print("="*80)

# def save_oracle_summary(dataset_key: str, results: Dict):
#     """
#     Сохранение сводной статистики по оракулам.
#     """
#     summary = {
#         'dataset': dataset_key,
#         'model': TRANSFORMER_MODEL_NAME,
#         'gigachat_available': GIGACHAT_AVAILABLE,
#         'diversity_config': {
#             'use_diversity_init': DIV_CFG.use_diversity_init,
#             'diversity_samples': DIV_CFG.diversity_samples,
#             'oracle_types': list(results.keys()),
#             'hybrid_k': DIV_CFG.hybrid_k,
#         },
#         'results': {}
#     }

#     for oracle_type, oracle_results in results.items():
#         summary['results'][oracle_type] = {}

#         for strategy, strategy_results in oracle_results.items():
#             if 'aggregated' not in strategy_results or not strategy_results['aggregated']:
#                 continue

#             df = pd.DataFrame(strategy_results['aggregated'])

#             if df.empty:
#                 continue

#             # Находим максимальную accuracy
#             max_acc_idx = df['acc_mean'].idxmax()

#             # Вычисляем AUC
#             auc = np.trapz(df['acc_mean'], df['labeled'])

#             summary['results'][oracle_type][strategy] = {
#                 'best_accuracy': {
#                     'value': float(df.loc[max_acc_idx, 'acc_mean']),
#                     'std': float(df.loc[max_acc_idx, 'acc_std']),
#                     'labeled_samples': int(df.loc[max_acc_idx, 'labeled']),
#                     'cost': float(df.loc[max_acc_idx, 'total_cost_mean']),
#                 },
#                 'final_accuracy': {
#                     'value': float(df.iloc[-1]['acc_mean']),
#                     'std': float(df.iloc[-1]['acc_std']),
#                     'labeled_samples': int(df.iloc[-1]['labeled']),
#                     'cost': float(df.iloc[-1]['total_cost_mean']),
#                 },
#                 'area_under_curve': float(auc),
#                 'total_human_labels': float(df.iloc[-1]['cum_human_labels_mean']),
#                 'total_llm_labels': float(df.iloc[-1]['cum_llm_labels_mean']),
#                 'cached': strategy_results.get('cached', False)
#             }

#     # Сохранение сводки
#     summary_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_diversity_summary.json")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     print(f"\n Diversity summary saved: {summary_path}")

#     # Вывод таблицы
#     print(f"\n{'='*80}")
#     print(f"DIVERSITY SUMMARY FOR {dataset_key.upper()}:")
#     print('='*80)

#     for oracle_type in results.keys():
#         print(f"\nOracle: {oracle_type.upper()}")
#         print("-" * 80)

#         if 'results' in summary and oracle_type in summary['results']:
#             for strategy in summary['results'][oracle_type]:
#                 data = summary['results'][oracle_type][strategy]
#                 print(f"{strategy.upper():<15} | "
#                       f"Best: {data['best_accuracy']['value']:.4f}±{data['best_accuracy']['std']:.4f} @{data['best_accuracy']['labeled_samples']} | "
#                       f"Final: {data['final_accuracy']['value']:.4f}±{data['final_accuracy']['std']:.4f} | "
#                       f"AUC: {data['area_under_curve']:.2f} | "
#                       f"Human/LLM: {data['total_human_labels']:.0f}/{data['total_llm_labels']:.0f}")

#     print('='*80)

# def setup_experiment_logger(experiment_name: str, dataset_key: str, strategy_name: str, seed: int):
#     """Настройка логгера для эксперимента"""
#     log_dir = Path(LOG_DIR) / "experiment_logs"
#     log_dir.mkdir(exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = log_dir / f"{experiment_name}_{timestamp}.log"

#     logger = logging.getLogger(f"{experiment_name}_{seed}")
#     logger.setLevel(logging.DEBUG)

#     # Очистка существующих обработчиков
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # Файловый обработчик
#     file_handler = logging.FileHandler(log_file, encoding='utf-8')
#     file_handler.setLevel(logging.DEBUG)

#     # Консольный обработчик
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)

#     # Форматирование
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger, log_file

# def log_experiment_config(logger, cfg: ALConfig, dataset_key: str, strategy_name: str):
#     """Логирование конфигурации эксперимента"""
#     logger.info("=" * 60)
#     logger.info("EXPERIMENT CONFIGURATION")
#     logger.info("=" * 60)
#     logger.info(f"Dataset: {dataset_key}")
#     logger.info(f"Strategy: {strategy_name}")
#     logger.info(f"Model: {TRANSFORMER_MODEL_NAME}")
#     logger.info(f"Pool size: {cfg.pool_size}")
#     logger.info(f"Initial labeled: {cfg.initial_labeled}")
#     logger.info(f"Batch size: {cfg.batch_size}")
#     logger.info(f"Max labeled: {cfg.max_labeled}")
#     logger.info(f"Iterations: {cfg.iterations}")
#     logger.info(f"Human cost: {cfg.cost_human}")
#     logger.info(f"LLM cost: {cfg.cost_llm}")
#     logger.info(f"Max length: {cfg.max_length}")
#     logger.info(f"Training epochs: {cfg.num_epochs}")
#     logger.info(f"Training batch size: {cfg.train_batch_size}")
#     logger.info(f"Device: {DEVICE}")
#     logger.info("=" * 60)

# def run_single_al_experiment(
#     dataset_key: str,
#     strategy_name: str,
#     seed: int,
#     cfg: ALConfig,
#     save_history: bool = True,
#     verbose: bool = True,
# ):
#     """Запуск одного эксперимента с заданным сидом с подробным логированием"""
#     start_time = datetime.now()

#     # Настройка логирования
#     experiment_name = f"{dataset_key}_{strategy_name}_seed{seed}"
#     logger, log_file = setup_experiment_logger(experiment_name, dataset_key, strategy_name, seed)

#     try:
#         # Логирование начала эксперимента
#         logger.info(f" STARTING EXPERIMENT: {experiment_name}")
#         logger.info(f"Start time: {start_time}")

#         # Установка сида
#         set_seed(seed)
#         logger.info(f"Random seed set to: {seed}")

#         # Загрузка данных
#         logger.info(f"Loading dataset: {dataset_key}")
#         ds_std = standardized_datasets[dataset_key]
#         train_hf = ds_std["train"]
#         eval_hf, eval_name = get_eval_split(ds_std)

#         num_classes = len(set(train_hf["label"]))
#         logger.info(f"Number of classes: {num_classes}")
#         logger.info(f"Evaluation split: {eval_name}")
#         logger.info(f"Training samples: {len(train_hf)}")
#         logger.info(f"Evaluation samples: {len(eval_hf)}")

#         # Логирование конфигурации
#         log_experiment_config(logger, cfg, dataset_key, strategy_name)

#         # Сэмплирование пула
#         logger.info(f"Subsampling pool of size {cfg.pool_size} from training data...")
#         pool_hf, pool_indices = subsample_pool(train_hf, cfg.pool_size, seed)
#         logger.info(f"Pool created: {len(pool_hf)} samples")

#         # Подготовка датасетов
#         logger.info("Converting to TransformersDataset format...")
#         pool_ds = to_transformers_dataset(pool_hf, num_classes, max_length=cfg.max_length)
#         test_ds = to_transformers_dataset(eval_hf, num_classes, max_length=cfg.max_length)

#         # Создание классификатора и стратегии
#         logger.info(f"Creating classifier factory for {num_classes} classes...")
#         clf_factory = make_classifier_factory(num_classes, cfg)

#         logger.info(f"Creating query strategy: {strategy_name}...")
#         query_strategy = make_query_strategy(strategy_name, num_classes)

#         # Инициализация Active Learner
#         logger.info("Initializing PoolBasedActiveLearner...")
#         active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, pool_ds)

#         # Балансированная инициализация
#         logger.info(f"Performing balanced initialization with {cfg.initial_labeled} samples...")
#         indices_initial = random_initialization_balanced(
#             pool_ds.y, n_samples=cfg.initial_labeled
#         )
#         y_initial = np.array([pool_ds.y[i] for i in indices_initial])

#         # Логирование начального распределения классов
#         initial_class_dist = np.bincount(y_initial, minlength=num_classes)
#         logger.info(f"Initial class distribution: {initial_class_dist.tolist()}")

#         active_learner.initialize_data(indices_initial, y_initial)
#         logger.info(f"Active learner initialized with {len(indices_initial)} labeled samples")

#         history = []
#         indices_labeled = indices_initial.copy()

#         logger.info("=" * 60)
#         logger.info("STARTING ACTIVE LEARNING ITERATIONS")
#         logger.info("=" * 60)

#         # AL итерации
#         for it in range(cfg.iterations + 1):
#             iter_start_time = datetime.now()

#             # Оценка модели
#             logger.info(f"[Iteration {it}] Evaluating model...")
#             metrics = evaluate_on_test(active_learner, test_ds)
#             labeled_count = len(indices_labeled)

#             total_cost_human = labeled_count * cfg.cost_human

#             # Сохранение метрик
#             step_info = dict(
#                 iter=it,
#                 seed=seed,
#                 labeled=labeled_count,
#                 acc=metrics["acc"],
#                 macro_f1=metrics["macro_f1"],
#                 per_class_f1=metrics.get("per_class_f1", []),
#                 cost_human=total_cost_human,
#                 timestamp=iter_start_time.isoformat(),
#             )
#             history.append(step_info)

#             # Логирование метрик итерации
#             iter_duration = (datetime.now() - iter_start_time).total_seconds()
#             logger.info(f"[Iteration {it}] Results:")
#             logger.info(f"  Labeled samples: {labeled_count}/{cfg.pool_size}")
#             logger.info(f"  Accuracy: {metrics['acc']:.4f}")
#             logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
#             logger.info(f"  Human cost: {total_cost_human:.1f}")
#             logger.info(f"  Iteration duration: {iter_duration:.2f}s")

#             if verbose and it % 5 == 0:
#                 print(f"[Seed {seed}, Iter {it:02d}] labeled={labeled_count:4d} | "
#                       f"acc={metrics['acc']:.4f} | f1={metrics['macro_f1']:.4f}")

#             # Проверка завершения
#             if it == cfg.iterations:
#                 logger.info(f"Reached maximum iterations ({cfg.iterations}). Stopping.")
#                 break

#             # Запрос новых примеров
#             logger.info(f"[Iteration {it}] Querying {cfg.batch_size} new samples...")
#             query_start = datetime.now()
#             queried_indices = active_learner.query(num_samples=cfg.batch_size)
#             query_duration = (datetime.now() - query_start).total_seconds()

#             # Получение меток
#             y_queried = pool_ds.y[queried_indices]

#             # Логирование запрошенных примеров
#             queried_class_dist = np.bincount(y_queried, minlength=num_classes)
#             logger.info(f"[Iteration {it}] Query results:")
#             logger.info(f"  Queried indices: {queried_indices[:5]}..." if len(queried_indices) > 5 else f"  Queried indices: {queried_indices}")
#             logger.info(f"  Queried class distribution: {queried_class_dist.tolist()}")
#             logger.info(f"  Query duration: {query_duration:.2f}s")

#             # Обновление модели
#             logger.info(f"[Iteration {it}] Updating model with new labels...")
#             update_start = datetime.now()
#             active_learner.update(y_queried)
#             update_duration = (datetime.now() - update_start).total_seconds()
#             logger.info(f"  Update duration: {update_duration:.2f}s")

#             # Обновление индексов
#             indices_labeled = np.concatenate([indices_labeled, queried_indices])

#             # Проверка покрытия пула
#             coverage = len(indices_labeled) / cfg.pool_size * 100
#             logger.info(f"[Iteration {it}] Pool coverage: {coverage:.1f}%")

#             logger.info(f"[Iteration {it}] Completed in {iter_duration:.2f}s")
#             logger.info("-" * 40)

#         # Итоговое логирование
#         total_duration = (datetime.now() - start_time).total_seconds()
#         final_labeled = len(indices_labeled)
#         final_accuracy = history[-1]["acc"] if history else 0.0

#         logger.info("=" * 60)
#         logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
#         logger.info("=" * 60)
#         logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
#         logger.info(f"Final labeled samples: {final_labeled}")
#         logger.info(f"Final accuracy: {final_accuracy:.4f}")
#         logger.info(f"Final macro F1: {history[-1]['macro_f1']:.4f}" if history else "N/A")
#         logger.info(f"Total human cost: {final_labeled * cfg.cost_human:.1f}")
#         logger.info(f"Log file: {log_file}")
#         logger.info("=" * 60)

#         # Сохранение истории
#         if save_history:
#             save_single_history(history, experiment_name)
#             logger.info(f"History saved for experiment: {experiment_name}")

#         return history

#     except Exception as e:
#         # Логирование ошибок
#         logger.error(f" EXPERIMENT FAILED: {experiment_name}")
#         logger.error(f"Error type: {type(e).__name__}")
#         logger.error(f"Error message: {str(e)}")
#         logger.error("Traceback:", exc_info=True)

#         # Сохранение частичной истории при ошибке
#         if 'history' in locals() and history:
#             error_experiment_name = f"{experiment_name}_ERROR"
#             save_single_history(history, error_experiment_name)
#             logger.info(f"Partial history saved as: {error_experiment_name}")

#         raise  # Проброс исключения дальше

#     finally:
#         # Очистка логгера
#         logger.handlers.clear()
#         logging.getLogger(f"{experiment_name}_{seed}").handlers.clear()

# def run_multiple_al_experiments(
#     dataset_key: str,
#     strategy_name: str,
#     seeds: List[int],
#     cfg: ALConfig = None,
#     verbose: bool = True,
# ):
#     """Запуск нескольких экспериментов с разными сидами и агрегация результатов"""
#     if cfg is None:
#         cfg = ALConfig()

#     all_histories = []
#     failed_runs = []

#     print(f"\n{'='*60}")
#     print(f"RUNNING {len(seeds)} EXPERIMENTS: {dataset_key.upper()} - {strategy_name.upper()}")
#     print('='*60)

#     for i, seed in enumerate(seeds, 1):
#         print(f"\n Run {i}/{len(seeds)} (Seed: {seed})")

#         try:
#             history = run_single_al_experiment(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy_name,
#                 seed=seed,
#                 cfg=cfg,
#                 save_history=True,
#                 verbose=verbose
#             )
#             all_histories.append(history)
#             print(f"Success - Final accuracy: {history[-1]['acc']:.4f}")

#         except Exception as e:
#             print(f"   Failed - Error: {str(e)}")
#             failed_runs.append((seed, str(e)))

#     # Статистика выполнения
#     print(f"\n{'='*60}")
#     print("EXECUTION SUMMARY")
#     print('='*60)
#     print(f"Total runs: {len(seeds)}")
#     print(f"Successful: {len(all_histories)}")
#     print(f"Failed: {len(failed_runs)}")

#     if failed_runs:
#         print("\nFailed runs:")
#         for seed, error in failed_runs:
#             print(f"  Seed {seed}: {error}")

#     if all_histories:
#         # Агрегация результатов
#         aggregated = aggregate_histories(all_histories)

#         # Сохранение агрегированных результатов
#         exp_name = f"{dataset_key}_{strategy_name}_aggregated"
#         save_aggregated_history(aggregated, exp_name)

#         # Логирование агрегированных результатов
#         agg_logger = logging.getLogger("aggregation")
#         agg_logger.setLevel(logging.INFO)

#         log_file = Path(AGGREGATED_LOG_DIR) / f"{exp_name}_summary.log"
#         file_handler = logging.FileHandler(log_file, encoding='utf-8')
#         file_handler.setLevel(logging.INFO)

#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         agg_logger.addHandler(file_handler)

#         # Статистика по агрегированным результатам
#         df = pd.DataFrame(aggregated)

#         agg_logger.info(f"Aggregation for {dataset_key} - {strategy_name}")
#         agg_logger.info(f"Number of successful runs: {len(all_histories)}")
#         agg_logger.info(f"Seeds used: {[s for s in seeds if s not in [f[0] for f in failed_runs]]}")

#         # Ключевые метрики
#         final_metrics = df.iloc[-1]
#         agg_logger.info(f"\nFinal metrics (at {final_metrics['labeled']} samples):")
#         agg_logger.info(f"  Mean accuracy: {final_metrics['acc_mean']:.4f}")
#         agg_logger.info(f"  Accuracy std: ±{final_metrics['acc_std']:.4f}")
#         agg_logger.info(f"  Accuracy range: [{final_metrics['acc_min']:.4f}, {final_metrics['acc_max']:.4f}]")
#         agg_logger.info(f"  Mean macro F1: {final_metrics['macro_f1_mean']:.4f}")
#         agg_logger.info(f"  Human cost: {final_metrics['cost_human_mean']:.1f}")

#         # Лучшая accuracy
#         best_idx = df['acc_mean'].idxmax()
#         best_metrics = df.loc[best_idx]
#         agg_logger.info(f"\nBest accuracy achieved at {best_metrics['labeled']} samples:")
#         agg_logger.info(f"  Accuracy: {best_metrics['acc_mean']:.4f} ± {best_metrics['acc_std']:.4f}")

#         agg_logger.handlers.clear()

#         return aggregated, all_histories, failed_runs
#     else:
#         print("No successful runs to aggregate.")
#         return None, [], failed_runs

# def aggregate_histories(all_histories: List[List[Dict]]) -> Dict[str, Any]:
#     """Агрегация нескольких запусков в средние значения и стандартные отклонения"""
#     # Преобразуем в DataFrame для удобства
#     dfs = []
#     for i, history in enumerate(all_histories):
#         df = pd.DataFrame(history)
#         df['run'] = i
#         dfs.append(df)

#     combined_df = pd.concat(dfs, ignore_index=True)

#     # Группируем по количеству размеченных объектов
#     aggregated_results = []

#     # Получаем все уникальные значения labeled
#     all_labeled = sorted(combined_df['labeled'].unique())

#     for labeled in all_labeled:
#         mask = combined_df['labeled'] == labeled
#         group = combined_df[mask]

#         if len(group) > 0:
#             agg_entry = {
#                 'labeled': int(labeled),
#                 'acc_mean': float(group['acc'].mean()),
#                 'acc_std': float(group['acc'].std()),
#                 'acc_min': float(group['acc'].min()),
#                 'acc_max': float(group['acc'].max()),
#                 'macro_f1_mean': float(group['macro_f1'].mean()),
#                 'macro_f1_std': float(group['macro_f1'].std()),
#                 'macro_f1_min': float(group['macro_f1'].min()),
#                 'macro_f1_max': float(group['macro_f1'].max()),
#                 'cost_human_mean': float(group['cost_human'].mean()),
#                 'cost_human_std': float(group['cost_human'].std()),
#                 'n_runs': int(len(group)),
#             }
#             aggregated_results.append(agg_entry)

#     return aggregated_results

# def save_single_history(history: List[Dict], experiment_name: str):
#     """Сохранение истории одного запуска"""
#     os.makedirs(LOG_DIR, exist_ok=True)
#     json_path = os.path.join(LOG_DIR, f"{experiment_name}_history.json")
#     csv_path = os.path.join(LOG_DIR, f"{experiment_name}_history.csv")

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)

#     df = pd.DataFrame(history)
#     df.to_csv(csv_path, index=False)
# def save_aggregated_history(aggregated: List[Dict], experiment_name: str, cfg: ALConfig = None):
#     """
#     Сохраняем агрегированные результаты на диск.
#     Теперь принимает cfg как необязательный параметр для обратной совместимости.
#     """
#     os.makedirs(AGGREGATED_LOG_DIR, exist_ok=True)
#     json_path = os.path.join(AGGREGATED_LOG_DIR, f"{experiment_name}.json")
#     csv_path = os.path.join(AGGREGATED_LOG_DIR, f"{experiment_name}.csv")

#     # Сохраняем результаты
#     if cfg is not None:
#         # Новый формат с конфигурацией
#         data_to_save = {
#             'config': {
#                 'pool_size': cfg.pool_size,
#                 'initial_labeled': cfg.initial_labeled,
#                 'batch_size': cfg.batch_size,
#                 'max_labeled': cfg.max_labeled,
#                 'iterations': cfg.iterations,
#                 'max_length': cfg.max_length,
#                 'num_epochs': cfg.num_epochs,
#                 'train_batch_size': cfg.train_batch_size,
#             },
#             'results': aggregated
#         }
#     else:
#         # Старый формат для обратной совместимости
#         data_to_save = aggregated

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(data_to_save, f, ensure_ascii=False, indent=2)

#     # Для CSV всегда сохраняем только результаты
#     df = pd.DataFrame(aggregated)
#     df.to_csv(csv_path, index=False)

#     print(f"Aggregated results saved to:\n  {json_path}\n  {csv_path}")
# def plot_aggregated_results(aggregated: List[Dict], title: str, dataset_name: str, strategy_name: str):
#     """Визуализация агрегированных результатов с доверительными интервалами"""
#     df = pd.DataFrame(aggregated)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # График 1: Accuracy vs количество размеченных объектов
#     ax1 = axes[0]
#     ax1.plot(df['labeled'], df['acc_mean'], 'b-', label='Mean', linewidth=2)
#     ax1.fill_between(df['labeled'],
#                      df['acc_mean'] - df['acc_std'],
#                      df['acc_mean'] + df['acc_std'],
#                      alpha=0.3, color='b', label='±1 std')
#     ax1.fill_between(df['labeled'],
#                      df['acc_min'],
#                      df['acc_max'],
#                      alpha=0.1, color='b', label='Min-Max')

#     ax1.set_xlabel('Number of Labeled Samples', fontsize=12)
#     ax1.set_ylabel('Accuracy', fontsize=12)
#     ax1.set_title(f'{title}\nAccuracy vs #Labeled Samples', fontsize=14)
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # График 2: Accuracy vs стоимость
#     ax2 = axes[1]
#     ax2.plot(df['cost_human_mean'], df['acc_mean'], 'r-', label='Mean', linewidth=2)
#     ax2.fill_between(df['cost_human_mean'],
#                      df['acc_mean'] - df['acc_std'],
#                      df['acc_mean'] + df['acc_std'],
#                      alpha=0.3, color='r', label='±1 std')

#     ax2.set_xlabel('Cost (Human Labeling)', fontsize=12)
#     ax2.set_ylabel('Accuracy', fontsize=12)
#     ax2.set_title(f'{title}\nAccuracy vs Cost', fontsize=14)
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()

#     plt.tight_layout()

#     # Сохранение графика
#     plot_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_name}_{strategy_name}_plot.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()

#     # Дополнительный график: детальная статистика
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(df['labeled'], df['acc_mean'],
#                  yerr=df['acc_std'],
#                  fmt='o-', capsize=5, capthick=2,
#                  label=f'{strategy_name} (N={df["n_runs"].iloc[0]})')

#     plt.xlabel('Number of Labeled Samples', fontsize=12)
#     plt.ylabel('Accuracy ± Std', fontsize=12)
#     plt.title(f'{dataset_name} - {strategy_name}\nAccuracy with Confidence Intervals', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()

#     detail_plot_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_name}_{strategy_name}_detail.png")
#     plt.savefig(detail_plot_path, dpi=300, bbox_inches='tight')
#     plt.show()

# def compare_strategies(dataset_key: str, strategies: List[str], seeds: List[int]):
#     """Сравнение нескольких стратегий на одном датасете"""
#     all_results = {}

#     for strategy in strategies:
#         print(f"\n{'='*60}")
#         print(f"Running {strategy} strategy on {dataset_key}")
#         print('='*60)

#         aggregated, individual_histories = run_multiple_al_experiments(
#             dataset_key=dataset_key,
#             strategy_name=strategy,
#             seeds=seeds
#         )

#         all_results[strategy] = {
#             'aggregated': aggregated,
#             'individual': individual_histories
#         }

#         # Визуализация для каждой стратегии
#         plot_aggregated_results(
#             aggregated,
#             title=f"{dataset_key.upper()} - {strategy.upper()}",
#             dataset_name=dataset_key,
#             strategy_name=strategy
#         )

#     # Сводный график сравнения стратегий
#     plot_strategy_comparison(dataset_key, all_results)

#     return all_results

# def plot_strategy_comparison(dataset_key: str, all_results: Dict):
#     """Сводный график сравнения всех стратегий"""
#     plt.figure(figsize=(12, 6))

#     colors = {'random': 'blue', 'least_conf': 'green', 'bald': 'red', 'badge': 'orange'}

#     for strategy, results in all_results.items():
#         df = pd.DataFrame(results['aggregated'])
#         color = colors.get(strategy, 'black')

#         plt.plot(df['labeled'], df['acc_mean'],
#                  color=color, label=strategy.upper(), linewidth=2)

#         plt.fill_between(df['labeled'],
#                          df['acc_mean'] - df['acc_std'],
#                          df['acc_mean'] + df['acc_std'],
#                          alpha=0.2, color=color)

#     plt.xlabel('Number of Labeled Samples', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.title(f'{dataset_key.upper()} - Comparison of Active Learning Strategies\n(Mean ± Standard Deviation)',
#               fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title='Strategy')
#     plt.tight_layout()

#     comparison_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_strategy_comparison.png")
#     plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
#     plt.show()

# def log_dataset_statistics(dataset_key: str):
#     """Логирование статистики датасета"""
#     if dataset_key not in standardized_datasets:
#         print(f"Dataset {dataset_key} not found in standardized_datasets")
#         return

#     ds = standardized_datasets[dataset_key]
#     train = ds["train"]

#     print(f"\n{'='*60}")
#     print(f"DATASET STATISTICS: {dataset_key.upper()}")
#     print('='*60)

#     # Базовая статистика
#     print(f"Training samples: {len(train)}")

#     # Распределение классов
#     labels = [ex["label"] for ex in train]
#     label_counts = Counter(labels)
#     print(f"Classes: {len(label_counts)}")
#     print(f"Class distribution: {dict(label_counts)}")

#     # Статистика по длине текстов
#     lengths = [len(str(ex["text"]).split()) for ex in train]
#     print(f"Average text length: {np.mean(lengths):.1f} words")
#     print(f"Median text length: {np.median(lengths):.1f} words")
#     print(f"Min text length: {np.min(lengths)} words")
#     print(f"Max text length: {np.max(lengths)} words")
#     print(f"95th percentile: {np.percentile(lengths, 95):.1f} words")

#     # Примеры
#     print(f"\nSample texts (first 3):")
#     for i in range(min(3, len(train))):
#         text = train[i]["text"]
#         label = train[i]["label"]
#         preview = text[:100] + "..." if len(text) > 100 else text
#         print(f"  [{label}] {preview}")

# def log_model_info():
#     """Логирование информации о модели"""
#     print(f"\n{'='*60}")
#     print("MODEL INFORMATION")
#     print('='*60)
#     print(f"Model name: {TRANSFORMER_MODEL_NAME}")
#     print(f"Max sequence length: {MAX_LENGTH}")
#     print(f"Device: {DEVICE}")
#     print(f"CUDA available: {torch.cuda.is_available()}")

#     if torch.cuda.is_available():
#         print(f"GPU name: {torch.cuda.get_device_name(0)}")
#         print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# def create_experiment_report(
#     dataset_key: str,
#     strategy_name: str,
#     aggregated_results: List[Dict],
#     individual_histories: List[List[Dict]],
#     failed_runs: List[Tuple[int, str]],
#     cfg: ALConfig,
# ):
#     """Создание подробного отчета по эксперименту"""
#     report_dir = Path(AGGREGATED_LOG_DIR) / "reports"
#     report_dir.mkdir(exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     report_file = report_dir / f"{dataset_key}_{strategy_name}_{timestamp}_report.txt"

#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write("=" * 70 + "\n")
#         f.write("ACTIVE LEARNING EXPERIMENT REPORT\n")
#         f.write("=" * 70 + "\n\n")

#         # Метаданные
#         f.write("1. EXPERIMENT METADATA\n")
#         f.write("-" * 40 + "\n")
#         f.write(f"Dataset: {dataset_key}\n")
#         f.write(f"Strategy: {strategy_name}\n")
#         f.write(f"Model: {TRANSFORMER_MODEL_NAME}\n")
#         f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Report file: {report_file}\n\n")

#         # Конфигурация
#         f.write("2. CONFIGURATION\n")
#         f.write("-" * 40 + "\n")
#         f.write(f"Pool size: {cfg.pool_size}\n")
#         f.write(f"Initial labeled: {cfg.initial_labeled}\n")
#         f.write(f"Batch size: {cfg.batch_size}\n")
#         f.write(f"Max labeled: {cfg.max_labeled}\n")
#         f.write(f"Iterations: {cfg.iterations}\n")
#         f.write(f"Human cost per sample: {cfg.cost_human}\n")
#         f.write(f"LLM cost per sample: {cfg.cost_llm}\n")
#         f.write(f"Max sequence length: {cfg.max_length}\n")
#         f.write(f"Training epochs: {cfg.num_epochs}\n")
#         f.write(f"Training batch size: {cfg.train_batch_size}\n")
#         f.write(f"Device: {DEVICE}\n\n")

#         # Статистика выполнения
#         f.write("3. EXECUTION STATISTICS\n")
#         f.write("-" * 40 + "\n")
#         f.write(f"Total runs attempted: {len(individual_histories) + len(failed_runs)}\n")
#         f.write(f"Successful runs: {len(individual_histories)}\n")
#         f.write(f"Failed runs: {len(failed_runs)}\n")
#         f.write(f"Success rate: {len(individual_histories)/(len(individual_histories) + len(failed_runs))*100:.1f}%\n\n")

#         if failed_runs:
#             f.write("Failed runs details:\n")
#             for seed, error in failed_runs:
#                 f.write(f"  Seed {seed}: {error}\n")
#             f.write("\n")

#         # Агрегированные результаты
#         if aggregated_results:
#             df = pd.DataFrame(aggregated_results)

#             f.write("4. AGGREGATED RESULTS\n")
#             f.write("-" * 40 + "\n")

#             # Ключевые точки
#             key_points = [10, 50, 100, 150, 200, 250, 300]
#             f.write("Performance at key points:\n")
#             f.write("Labeled | Accuracy (mean±std) | Macro F1 (mean±std) | Cost\n")
#             f.write("-" * 70 + "\n")

#             for point in key_points:
#                 if point <= cfg.max_labeled:
#                     closest_idx = (df['labeled'] - point).abs().idxmin()
#                     row = df.loc[closest_idx]
#                     f.write(f"{row['labeled']:6d} | "
#                            f"{row['acc_mean']:.4f} ± {row['acc_std']:.4f} | "
#                            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
#                            f"{row['cost_human_mean']:.1f}\n")
#             f.write("\n")

#             # Итоговые метрики
#             final_row = df.iloc[-1]
#             f.write("Final performance:\n")
#             f.write(f"Labeled samples: {final_row['labeled']}\n")
#             f.write(f"Accuracy: {final_row['acc_mean']:.4f} ± {final_row['acc_std']:.4f}\n")
#             f.write(f"Accuracy range: [{final_row['acc_min']:.4f}, {final_row['acc_max']:.4f}]\n")
#             f.write(f"Macro F1: {final_row['macro_f1_mean']:.4f} ± {final_row['macro_f1_std']:.4f}\n")
#             f.write(f"Total cost: {final_row['cost_human_mean']:.1f}\n\n")

#             # Тренды
#             f.write("5. PERFORMANCE TRENDS\n")
#             f.write("-" * 40 + "\n")

#             # Ранний прогресс (первые 50 образцов)
#             early_df = df[df['labeled'] <= 50]
#             if len(early_df) > 0:
#                 early_improvement = early_df['acc_mean'].iloc[-1] - early_df['acc_mean'].iloc[0]
#                 f.write(f"Early progress (0-50 samples): +{early_improvement:.4f} accuracy\n")

#             # Поздний прогресс (последние 50 образцов)
#             late_df = df[df['labeled'] >= cfg.max_labeled - 50]
#             if len(late_df) > 0:
#                 late_improvement = late_df['acc_mean'].iloc[-1] - late_df['acc_mean'].iloc[0]
#                 f.write(f"Late progress (last 50 samples): +{late_improvement:.4f} accuracy\n")

#             # Точка уменьшения отдачи
#             acc_gains = np.diff(df['acc_mean'])
#             if len(acc_gains) > 0:
#                 diminishing_return_idx = np.where(acc_gains < np.mean(acc_gains) / 2)[0]
#                 if len(diminishing_return_idx) > 0:
#                     point = df['labeled'].iloc[diminishing_return_idx[0]]
#                     f.write(f"Diminishing returns start around: {point} samples\n")

#         f.write("\n" + "=" * 70 + "\n")
#         f.write("END OF REPORT\n")
#         f.write("=" * 70 + "\n")

#     print(f" Experiment report saved to: {report_file}")
#     return report_file

# def compare_strategies_with_logging(dataset_key: str, strategies: List[str], seeds: List[int], cfg: ALConfig = None):
#     """Сравнение нескольких стратегий на одном датасете с подробным логированием"""
#     if cfg is None:
#         cfg = ALConfig()

#     all_results = {}

#     print(f"\n{'#'*80}")
#     print(f"COMPREHENSIVE EXPERIMENT: {dataset_key.upper()}")
#     print('#'*80)

#     # Логирование информации о датасете и модели
#     log_dataset_statistics(dataset_key)
#     log_model_info()

#     for strategy in strategies:
#         print(f"\n{'='*80}")
#         print(f"STRATEGY: {strategy.upper()}")
#         print('='*80)

#         aggregated, individual_histories, failed_runs = run_multiple_al_experiments(
#             dataset_key=dataset_key,
#             strategy_name=strategy,
#             seeds=seeds,
#             cfg=cfg
#         )

#         if aggregated:
#             all_results[strategy] = {
#                 'aggregated': aggregated,
#                 'individual': individual_histories,
#                 'failed': failed_runs
#             }

#             # Создание отчета
#             report_file = create_experiment_report(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy,
#                 aggregated_results=aggregated,
#                 individual_histories=individual_histories,
#                 failed_runs=failed_runs,
#                 cfg=cfg
#             )

#             # Визуализация
#             plot_aggregated_results(
#                 aggregated,
#                 title=f"{dataset_key.upper()} - {strategy.upper()}",
#                 dataset_name=dataset_key,
#                 strategy_name=strategy
#             )

#     if all_results:
#         # Сводный график сравнения стратегий
#         plot_strategy_comparison(dataset_key, all_results)

#         # Сводный отчет по всем стратегиям
#         create_comparison_report(dataset_key, all_results, cfg)

#     return all_results


# def load_cached_results(
#     dataset_key: str,
#     strategy_name: str,
#     seeds: List[int]
# ) -> Tuple[Optional[List[Dict]], Optional[List[List[Dict]]], List[int]]:
#     """
#     Загружает закешированные результаты из файлов.
#     Возвращает (aggregated_data, individual_histories, missing_seeds)
#     """
#     aggregated_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_{strategy_name}_aggregated.json")

#     # Пытаемся загрузить агрегированные результаты
#     aggregated_data = None
#     if os.path.exists(aggregated_path):
#         try:
#             with open(aggregated_path, 'r', encoding='utf-8') as f:
#                 aggregated_data = json.load(f)
#             # Проверяем формат данных
#             if isinstance(aggregated_data, dict) and 'results' in aggregated_data:
#                 # Новый формат с конфигурацией
#                 aggregated_data = aggregated_data['results']
#             print(f"Загружены агрегированные результаты из кэша")
#         except Exception as e:
#             print(f"  Ошибка загрузки агрегированных результатов: {e}")
#             aggregated_data = None

#     # Загружаем индивидуальные истории
#     individual_histories = []
#     missing_seeds = []

#     for seed in seeds:
#         individual_path = os.path.join(LOG_DIR, f"{dataset_key}_{strategy_name}_seed{seed}_history.csv")

#         if os.path.exists(individual_path):
#             try:
#                 df = pd.read_csv(individual_path)
#                 history = df.to_dict('records')

#                 # Добавляем seed к каждой записи истории
#                 for record in history:
#                     record['seed'] = seed

#                 individual_histories.append(history)
#                 print(f"    Seed {seed}: загружено из кэша")
#             except Exception as e:
#                 print(f"    Seed {seed}: ошибка загрузки - {e}")
#                 missing_seeds.append(seed)
#         else:
#             missing_seeds.append(seed)

#     return aggregated_data, individual_histories, missing_seeds

# def check_config_compatibility(current_config: ALConfig) -> bool:
#     """
#     Проверяет совместимость конфигурации.
#     В текущей реализации просто возвращает True, так как мы используем фиксированную конфигурацию.
#     Можно расширить для проверки конкретных параметров.
#     """
#     # Здесь можно добавить проверку конкретных параметров
#     # Например, проверить, что модель, диапазон и другие параметры совпадают
#     return True

# def compare_strategies(
#     dataset_key: str,
#     strategies: List[str],
#     seeds: List[int],
#     cfg: ALConfig = None,
#     use_cache: bool = True,
#     force_rerun: bool = False
# ) -> Dict[str, Dict]:
#     """
#     Сравнение нескольких стратегий на одном датасете с поддержкой кэширования.
#     """
#     if cfg is None:
#         cfg = ALConfig()

#     all_results = {}

#     print(f"\n{'#'*80}")
#     print(f"COMPARING STRATEGIES FOR DATASET: {dataset_key.upper()}")
#     print(f"Strategies: {', '.join(strategies)}")
#     print(f"Seeds: {seeds}")
#     print(f"Use cache: {use_cache}")
#     print(f"Force rerun: {force_rerun}")
#     print('#'*80)

#     for strategy in strategies:
#         print(f"\n{'='*80}")
#         print(f"PROCESSING STRATEGY: {strategy.upper()}")
#         print('='*80)

#         aggregated_data = None
#         individual_histories = []
#         failed_runs = []

#         # Пытаемся загрузить из кэша
#         if use_cache and not force_rerun:
#             print(" Проверяем кэш...")
#             cached_aggregated, cached_individual, missing_seeds = load_cached_results(
#                 dataset_key, strategy, seeds
#             )

#             if cached_aggregated is not None and cached_individual:
#                 # Проверяем совместимость конфигурации
#                 config_compatible = check_config_compatibility(cfg)

#                 if config_compatible and len(cached_individual) >= len(seeds) * 0.8:  # хотя бы 80% сидов
#                     print(f"Загружено из кэша ({len(cached_individual)} из {len(seeds)} сидов)")
#                     aggregated_data = cached_aggregated
#                     individual_histories = cached_individual

#                     # Создаем объекты DataFrame для индивидуальных историй
#                     individual_dfs = []
#                     for history in individual_histories:
#                         df = pd.DataFrame(history)
#                         individual_dfs.append(df)

#                     # Визуализируем кэшированные результаты
#                     if aggregated_data:
#                         plot_aggregated_results(
#                             aggregated_data,
#                             title=f"{dataset_key.upper()} - {strategy.upper()} (CACHED)",
#                             dataset_name=dataset_key,
#                             strategy_name=strategy
#                         )

#                     all_results[strategy] = {
#                         'aggregated': aggregated_data,
#                         'individual': individual_dfs,
#                         'failed': failed_runs,
#                         'cached': True
#                     }

#                     # Если все сиды загружены, переходим к следующей стратегии
#                     if len(cached_individual) == len(seeds):
#                         continue
#                     else:
#                         print(f"  Загружено не все, запускаем недостающие сиды")
#                         seeds_to_run = missing_seeds
#                 else:
#                     print(f"  Кэш неполный или несовместим, запускаем все сиды")
#                     seeds_to_run = seeds
#             else:
#                 print(f"  Кэш не найден или неполный, запускаем все сиды")
#                 seeds_to_run = seeds
#         else:
#             seeds_to_run = seeds

#         # Запускаем эксперименты для недостающих сидов
#         if seeds_to_run:
#             print(f" Запускаем эксперименты для сидов: {seeds_to_run}")

#             aggregated, individual, failed = run_multiple_al_experiments(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy,
#                 seeds=seeds_to_run,
#                 cfg=cfg,
#                 verbose=True
#             )

#             if aggregated:
#                 # Если у нас уже были кэшированные данные, объединяем их
#                 if aggregated_data and individual_histories:
#                     # Объединяем агрегированные данные
#                     old_df = pd.DataFrame(aggregated_data)
#                     new_df = pd.DataFrame(aggregated)

#                     # Объединяем по количеству размеченных объектов
#                     combined_df = pd.concat([old_df, new_df]).groupby('labeled').agg({
#                         'acc_mean': 'mean',
#                         'acc_std': lambda x: np.sqrt(np.mean(np.square(x))),  # объединенное std
#                         'acc_min': 'min',
#                         'acc_max': 'max',
#                         'macro_f1_mean': 'mean',
#                         'macro_f1_std': lambda x: np.sqrt(np.mean(np.square(x))),
#                         'macro_f1_min': 'min',
#                         'macro_f1_max': 'max',
#                         'cost_human_mean': 'mean',
#                         'cost_human_std': lambda x: np.sqrt(np.mean(np.square(x))),
#                         'n_runs': 'sum'
#                     }).reset_index()

#                     aggregated_data = combined_df.to_dict('records')

#                     # Объединяем индивидуальные истории
#                     individual_histories.extend(individual)
#                 else:
#                     aggregated_data = aggregated
#                     individual_histories = individual

#                 failed_runs.extend(failed)

#                 # Визуализируем новые результаты
#                 plot_aggregated_results(
#                     aggregated_data,
#                     title=f"{dataset_key.upper()} - {strategy.upper()}",
#                     dataset_name=dataset_key,
#                     strategy_name=strategy
#                 )
#             else:
#                 print(f"   Не удалось получить агрегированные результаты для стратегии {strategy}")

#         # Сохраняем результаты
#         if aggregated_data:
#             all_results[strategy] = {
#                 'aggregated': aggregated_data,
#                 'individual': individual_histories,
#                 'failed': failed_runs,
#                 'cached': use_cache and not force_rerun
#             }
#         else:
#             print(f"  Нет данных для стратегии {strategy}")

#     return all_results

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

# def save_summary_statistics(dataset_key: str, results: Dict):
#     """Сохранение сводной статистики по всем стратегиям"""
#     summary = {
#         'dataset': dataset_key,
#         'model': TRANSFORMER_MODEL_NAME,
#         'num_runs': NUM_RUNS,
#         'seeds_used': SEEDS[:NUM_RUNS],
#         'timestamp': datetime.now().isoformat(),
#         'config': {
#             'pool_size': POOL_SIZE,
#             'initial_labeled': INITIAL_LABELED,
#             'batch_size': BATCH_SIZE,
#             'max_labeled': MAX_LABELED,
#             'iterations': N_ITER,
#             'cost_human': COST_HUMAN,
#             'cost_llm': COST_LLM,
#             'max_length': MAX_LENGTH,
#             'num_epochs': 3,
#             'train_batch_size': 16,
#         },
#         'results': {}
#     }

#     for strategy, strategy_results in results.items():
#         if 'aggregated' not in strategy_results or not strategy_results['aggregated']:
#             continue

#         df = pd.DataFrame(strategy_results['aggregated'])

#         if df.empty:
#             continue

#         # Находим максимальную accuracy и соответствующие метрики
#         max_acc_idx = df['acc_mean'].idxmax()

#         # Вычисляем AUC
#         auc = np.trapz(df['acc_mean'], df['labeled'])

#         # Анализ сходимости
#         early_perf = df[df['labeled'] <= 100]['acc_mean'].mean() if len(df[df['labeled'] <= 100]) > 0 else 0
#         late_perf = df[df['labeled'] >= 200]['acc_mean'].mean() if len(df[df['labeled'] >= 200]) > 0 else 0

#         summary['results'][strategy] = {
#             'best_accuracy': {
#                 'value': float(df.loc[max_acc_idx, 'acc_mean']),
#                 'std': float(df.loc[max_acc_idx, 'acc_std']),
#                 'min': float(df.loc[max_acc_idx, 'acc_min']),
#                 'max': float(df.loc[max_acc_idx, 'acc_max']),
#                 'labeled_samples': int(df.loc[max_acc_idx, 'labeled']),
#                 'cost': float(df.loc[max_acc_idx, 'cost_human_mean'])
#             },
#             'final_accuracy': {
#                 'value': float(df.iloc[-1]['acc_mean']),
#                 'std': float(df.iloc[-1]['acc_std']),
#                 'min': float(df.iloc[-1]['acc_min']),
#                 'max': float(df.iloc[-1]['acc_max']),
#                 'labeled_samples': int(df.iloc[-1]['labeled']),
#                 'cost': float(df.iloc[-1]['cost_human_mean'])
#             },
#             'area_under_curve': float(auc),
#             'early_performance': float(early_perf),
#             'late_performance': float(late_perf),
#             'improvement': float(late_perf - early_perf),
#             'cached': strategy_results.get('cached', False),
#             'failed_runs': len(strategy_results.get('failed', []))
#         }

#     # Сохранение сводки
#     summary_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_summary.json")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     print(f"\n Сводная статистика сохранена: {summary_path}")

#     # Вывод сводной таблицы в консоль
#     print(f"\n{'='*80}")
#     print(f"SUMMARY FOR {dataset_key.upper()}:")
#     print('='*80)

#     if summary['results']:
#         summary_df = pd.DataFrame({
#             'Strategy': [],
#             'Best Acc': [],
#             '±Std': [],
#             '@N': [],
#             'Final Acc': [],
#             '±Std': [],
#             'AUC': [],
#             'Cached': [],
#             'Failed': []
#         })

#         for strategy, strat_results in summary['results'].items():
#             best = strat_results['best_accuracy']
#             final = strat_results['final_accuracy']

#             summary_df.loc[len(summary_df)] = [
#                 strategy.upper(),
#                 f"{best['value']:.4f}",
#                 f"±{best['std']:.4f}",
#                 best['labeled_samples'],
#                 f"{final['value']:.4f}",
#                 f"±{final['std']:.4f}",
#                 f"{strat_results['area_under_curve']:.2f}",
#                 "✓" if strat_results['cached'] else "✗",
#                 strat_results['failed_runs']
#             ]

#         print(summary_df.to_string(index=False))
#     else:
#         print("No results available")

#     print('='*80)


# def load_cached_results(
#     dataset_key: str,
#     strategy_name: str,
#     seeds: List[int]
# ) -> Tuple[Optional[List[Dict]], Optional[List[List[Dict]]], List[int]]:
#     """
#     Загружает закешированные результаты из файлов.
#     Возвращает (aggregated_data, individual_histories, missing_seeds)
#     """
#     aggregated_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_{strategy_name}_aggregated.json")

#     # Пытаемся загрузить агрегированные результаты
#     aggregated_data = None
#     if os.path.exists(aggregated_path):
#         try:
#             with open(aggregated_path, 'r', encoding='utf-8') as f:
#                 aggregated_data = json.load(f)
#             # Проверяем формат данных
#             if isinstance(aggregated_data, dict) and 'results' in aggregated_data:
#                 # Новый формат с конфигурацией
#                 aggregated_data = aggregated_data['results']
#             print(f"Загружены агрегированные результаты из кэша")
#         except Exception as e:
#             print(f"  Ошибка загрузки агрегированных результатов: {e}")
#             aggregated_data = None

#     # Загружаем индивидуальные истории
#     individual_histories = []
#     missing_seeds = []

#     for seed in seeds:
#         individual_path = os.path.join(LOG_DIR, f"{dataset_key}_{strategy_name}_seed{seed}_history.csv")

#         if os.path.exists(individual_path):
#             try:
#                 df = pd.read_csv(individual_path)
#                 history = df.to_dict('records')

#                 # Добавляем seed к каждой записи истории
#                 for record in history:
#                     record['seed'] = seed

#                 individual_histories.append(history)
#                 print(f"    Seed {seed}: загружено из кэша")
#             except Exception as e:
#                 print(f"    Seed {seed}: ошибка загрузки - {e}")
#                 missing_seeds.append(seed)
#         else:
#             missing_seeds.append(seed)

#     return aggregated_data, individual_histories, missing_seeds

# def check_config_compatibility(current_config: ALConfig) -> bool:
#     """
#     Проверяет совместимость конфигурации.
#     В текущей реализации просто возвращает True, так как мы используем фиксированную конфигурацию.
#     """
#     return True

# def compare_strategies(
#     dataset_key: str,
#     strategies: List[str],
#     seeds: List[int],
#     cfg: ALConfig = None,
#     use_cache: bool = True,
#     force_rerun: bool = False
# ) -> Dict[str, Dict]:
#     """
#     Сравнение нескольких стратегий на одном датасете с поддержкой кэширования.
#     """
#     if cfg is None:
#         cfg = ALConfig()

#     all_results = {}

#     print(f"\n{'#'*80}")
#     print(f"COMPARING STRATEGIES FOR DATASET: {dataset_key.upper()}")
#     print(f"Strategies: {', '.join(strategies)}")
#     print(f"Seeds: {seeds}")
#     print(f"Use cache: {use_cache}")
#     print(f"Force rerun: {force_rerun}")
#     print('#'*80)

#     for strategy in strategies:
#         print(f"\n{'='*80}")
#         print(f"PROCESSING STRATEGY: {strategy.upper()}")
#         print('='*80)

#         aggregated_data = None
#         individual_histories = []
#         failed_runs = []

#         # Пытаемся загрузить из кэша
#         if use_cache and not force_rerun:
#             print(" Проверяем кэш...")
#             cached_aggregated, cached_individual, missing_seeds = load_cached_results(
#                 dataset_key, strategy, seeds
#             )

#             if cached_aggregated is not None and cached_individual:
#                 # Проверяем совместимость конфигурации
#                 config_compatible = check_config_compatibility(cfg)

#                 if config_compatible and len(cached_individual) >= len(seeds) * 0.8:  # хотя бы 80% сидов
#                     print(f"Загружено из кэша ({len(cached_individual)} из {len(seeds)} сидов)")
#                     aggregated_data = cached_aggregated
#                     individual_histories = cached_individual

#                     # Создаем объекты DataFrame для индивидуальных историй
#                     individual_dfs = []
#                     for history in individual_histories:
#                         df = pd.DataFrame(history)
#                         individual_dfs.append(df)

#                     # Визуализируем кэшированные результаты
#                     if aggregated_data and 'plot_aggregated_results' in globals():
#                         plot_aggregated_results(
#                             aggregated_data,
#                             title=f"{dataset_key.upper()} - {strategy.upper()} (CACHED)",
#                             dataset_name=dataset_key,
#                             strategy_name=strategy
#                         )

#                     all_results[strategy] = {
#                         'aggregated': aggregated_data,
#                         'individual': individual_dfs,
#                         'failed': failed_runs,
#                         'cached': True
#                     }

#                     # Если все сиды загружены, переходим к следующей стратегии
#                     if len(cached_individual) == len(seeds):
#                         continue
#                     else:
#                         print(f"  Загружено не все, запускаем недостающие сиды")
#                         seeds_to_run = missing_seeds
#                 else:
#                     print(f"  Кэш неполный или несовместим, запускаем все сиды")
#                     seeds_to_run = seeds
#             else:
#                 print(f"  Кэш не найден или неполный, запускаем все сиды")
#                 seeds_to_run = seeds
#         else:
#             seeds_to_run = seeds

#         # Запускаем эксперименты для недостающих сидов
#         if seeds_to_run:
#             print(f" Запускаем эксперименты для сидов: {seeds_to_run}")

#             # Импортируем здесь, чтобы избежать циклических импортов
#             try:
#                 from export_part3 import run_multiple_al_experiments
#             except ImportError:
#                 # Если функция не доступна, пропускаем
#                 print(f"  Функция run_multiple_al_experiments не доступна")
#                 continue

#             aggregated, individual, failed = run_multiple_al_experiments(
#                 dataset_key=dataset_key,
#                 strategy_name=strategy,
#                 seeds=seeds_to_run,
#                 cfg=cfg,
#                 verbose=True
#             )

#             if aggregated:
#                 # Если у нас уже были кэшированные данные, объединяем их
#                 if aggregated_data and individual_histories:
#                     # Объединяем агрегированные данные
#                     old_df = pd.DataFrame(aggregated_data)
#                     new_df = pd.DataFrame(aggregated)

#                     # Объединяем по количеству размеченных объектов
#                     combined_df = pd.concat([old_df, new_df]).groupby('labeled').agg({
#                         'acc_mean': 'mean',
#                         'acc_std': lambda x: np.sqrt(np.mean(np.square(x))),  # объединенное std
#                         'acc_min': 'min',
#                         'acc_max': 'max',
#                         'macro_f1_mean': 'mean',
#                         'macro_f1_std': lambda x: np.sqrt(np.mean(np.square(x))),
#                         'macro_f1_min': 'min',
#                         'macro_f1_max': 'max',
#                         'cost_human_mean': 'mean',
#                         'cost_human_std': lambda x: np.sqrt(np.mean(np.square(x))),
#                         'n_runs': 'sum'
#                     }).reset_index()

#                     aggregated_data = combined_df.to_dict('records')

#                     # Объединяем индивидуальные истории
#                     individual_histories.extend(individual)
#                 else:
#                     aggregated_data = aggregated
#                     individual_histories = individual

#                 failed_runs.extend(failed)

#                 # Визуализируем новые результаты
#                 if 'plot_aggregated_results' in globals():
#                     plot_aggregated_results(
#                         aggregated_data,
#                         title=f"{dataset_key.upper()} - {strategy.upper()}",
#                         dataset_name=dataset_key,
#                         strategy_name=strategy
#                     )
#             else:
#                 print(f"   Не удалось получить агрегированные результаты для стратегии {strategy}")

#         # Сохраняем результаты
#         if aggregated_data:
#             all_results[strategy] = {
#                 'aggregated': aggregated_data,
#                 'individual': individual_histories,
#                 'failed': failed_runs,
#                 'cached': use_cache and not force_rerun
#             }
#         else:
#             print(f"  Нет данных для стратегии {strategy}")

#     return all_results


# def save_summary_statistics(dataset_key: str, results: Dict):
#     """Сохранение сводной статистики по всем стратегиям"""
#     summary = {
#         'dataset': dataset_key,
#         'model': TRANSFORMER_MODEL_NAME,
#         'num_runs': NUM_RUNS,
#         'seeds_used': SEEDS[:NUM_RUNS],
#         'timestamp': datetime.now().isoformat(),
#         'config': {
#             'pool_size': POOL_SIZE,
#             'initial_labeled': INITIAL_LABELED,
#             'batch_size': BATCH_SIZE,
#             'max_labeled': MAX_LABELED,
#             'iterations': N_ITER,
#             'cost_human': COST_HUMAN,
#             'cost_llm': COST_LLM,
#             'max_length': MAX_LENGTH,
#             'num_epochs': 3,
#             'train_batch_size': 16,
#         },
#         'results': {}
#     }

#     for strategy, strategy_results in results.items():
#         if 'aggregated' not in strategy_results or not strategy_results['aggregated']:
#             continue

#         # Преобразуем в DataFrame
#         if isinstance(strategy_results['aggregated'], list):
#             df = pd.DataFrame(strategy_results['aggregated'])
#         else:
#             # Если это уже DataFrame
#             df = strategy_results['aggregated']

#         if df.empty:
#             continue

#         # Находим максимальную accuracy и соответствующие метрики
#         if 'acc_mean' in df.columns:
#             max_acc_idx = df['acc_mean'].idxmax()
#         else:
#             # Ищем альтернативные названия столбцов
#             acc_col = next((col for col in df.columns if 'acc' in col.lower()), None)
#             if acc_col:
#                 max_acc_idx = df[acc_col].idxmax()
#             else:
#                 continue

#         # Вычисляем AUC
#         if 'acc_mean' in df.columns and 'labeled' in df.columns:
#             auc = np.trapz(df['acc_mean'], df['labeled'])
#         else:
#             auc = 0

#         # Анализ сходимости
#         if 'acc_mean' in df.columns and 'labeled' in df.columns:
#             early_mask = df['labeled'] <= 100
#             late_mask = df['labeled'] >= 200

#             early_perf = df.loc[early_mask, 'acc_mean'].mean() if len(df[early_mask]) > 0 else 0
#             late_perf = df.loc[late_mask, 'acc_mean'].mean() if len(df[late_mask]) > 0 else 0
#         else:
#             early_perf = 0
#             late_perf = 0

#         # Получаем значения стоимости
#         cost_columns = [col for col in df.columns if 'cost' in col.lower()]
#         best_cost_col = 'cost_human_mean' if 'cost_human_mean' in df.columns else (cost_columns[0] if cost_columns else None)
#         final_cost_col = best_cost_col

#         best_cost = float(df.loc[max_acc_idx, best_cost_col]) if best_cost_col else 0.0
#         final_cost = float(df.iloc[-1][final_cost_col]) if final_cost_col else 0.0

#         summary['results'][strategy] = {
#             'best_accuracy': {
#                 'value': float(df.loc[max_acc_idx, 'acc_mean']) if 'acc_mean' in df.columns else 0,
#                 'std': float(df.loc[max_acc_idx, 'acc_std']) if 'acc_std' in df.columns else 0,
#                 'min': float(df.loc[max_acc_idx, 'acc_min']) if 'acc_min' in df.columns else 0,
#                 'max': float(df.loc[max_acc_idx, 'acc_max']) if 'acc_max' in df.columns else 0,
#                 'labeled_samples': int(df.loc[max_acc_idx, 'labeled']) if 'labeled' in df.columns else 0,
#                 'cost': best_cost
#             },
#             'final_accuracy': {
#                 'value': float(df.iloc[-1]['acc_mean']) if 'acc_mean' in df.columns else 0,
#                 'std': float(df.iloc[-1]['acc_std']) if 'acc_std' in df.columns else 0,
#                 'min': float(df.iloc[-1]['acc_min']) if 'acc_min' in df.columns else 0,
#                 'max': float(df.iloc[-1]['acc_max']) if 'acc_max' in df.columns else 0,
#                 'labeled_samples': int(df.iloc[-1]['labeled']) if 'labeled' in df.columns else 0,
#                 'cost': final_cost
#             },
#             'area_under_curve': float(auc),
#             'early_performance': float(early_perf),
#             'late_performance': float(late_perf),
#             'improvement': float(late_perf - early_perf),
#             'cached': strategy_results.get('cached', False),
#             'failed_runs': len(strategy_results.get('failed', []))
#         }

#     # Сохранение сводки
#     summary_path = os.path.join(AGGREGATED_LOG_DIR, f"{dataset_key}_summary.json")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     print(f"\n Сводная статистика сохранена: {summary_path}")

#     # Вывод сводной таблицы в консоль
#     print(f"\n{'='*80}")
#     print(f"SUMMARY FOR {dataset_key.upper()}:")
#     print('='*80)

#     if summary['results']:
#         # Создаем список строк для DataFrame
#         rows = []
#         for strategy, strat_results in summary['results'].items():
#             best = strat_results['best_accuracy']
#             final = strat_results['final_accuracy']

#             rows.append({
#                 'Strategy': strategy.upper(),
#                 'Best Acc': f"{best['value']:.4f}",
#                 'Best ±Std': f"±{best['std']:.4f}",
#                 '@N': best['labeled_samples'],
#                 'Final Acc': f"{final['value']:.4f}",
#                 'Final ±Std': f"±{final['std']:.4f}",
#                 'AUC': f"{strat_results['area_under_curve']:.2f}",
#                 'Cached': "✓" if strat_results['cached'] else "✗",
#                 'Failed': strat_results['failed_runs']
#             })

#         # Создаем DataFrame из списка словарей
#         summary_df = pd.DataFrame(rows)

#         # Переупорядочиваем столбцы для лучшего отображения
#         column_order = ['Strategy', 'Best Acc', 'Best ±Std', '@N', 'Final Acc', 'Final ±Std', 'AUC', 'Cached', 'Failed']
#         summary_df = summary_df[column_order]

#         print(summary_df.to_string(index=False))
#     else:
#         print("No results available")

#     print('='*80)

# def generate_final_report(all_datasets_results: Dict, cfg: ALConfig):
#     """Генерация итогового отчета по всем датасетам"""
#     report_dir = Path(AGGREGATED_LOG_DIR) / "final_reports"
#     report_dir.mkdir(exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     report_file = report_dir / f"final_report_{timestamp}.txt"

#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write("=" * 100 + "\n")
#         f.write("FINAL ACTIVE LEARNING EXPERIMENT REPORT\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 100 + "\n\n")

#         f.write("EXECUTIVE SUMMARY\n")
#         f.write("-" * 50 + "\n")
#         f.write(f"Model used: {TRANSFORMER_MODEL_NAME}\n")
#         f.write(f"Number of runs per experiment: {NUM_RUNS}\n")
#         f.write(f"Labeled samples range: {INITIAL_LABELED} to {MAX_LABELED}\n")
#         f.write(f"Batch size: {BATCH_SIZE}\n")
#         f.write(f"Datasets analyzed: {', '.join(DATASETS.keys())}\n")
#         f.write(f"Strategies compared: Random, Least Confidence, BALD, BADGE\n")
#         f.write(f"Total experiments: {len(DATASETS) * 4}\n\n")

#         # Результаты по каждому датасету
#         for dataset_key, results in all_datasets_results.items():
#             f.write(f"\n{'='*50}\n")
#             f.write(f"DATASET: {dataset_key.upper()}\n")
#             f.write('='*50 + "\n")

#             if not results:
#                 f.write("No results available\n")
#                 continue

#             # Находим лучшую стратегию по конечной точности
#             best_strategy = None
#             best_final_acc = 0

#             for strategy, strat_results in results.items():
#                 if 'aggregated' in strat_results and strat_results['aggregated']:
#                     if isinstance(strat_results['aggregated'], list):
#                         df = pd.DataFrame(strat_results['aggregated'])
#                     else:
#                         df = strat_results['aggregated']

#                     if not df.empty and 'acc_mean' in df.columns:
#                         final_acc = df.iloc[-1]['acc_mean']
#                         if final_acc > best_final_acc:
#                             best_final_acc = final_acc
#                             best_strategy = strategy

#             f.write(f"Best strategy: {best_strategy.upper() if best_strategy else 'N/A'}\n")
#             f.write(f"Best final accuracy: {best_final_acc:.4f}\n\n")

#             # Сводная таблица
#             f.write("Performance summary:\n")
#             f.write(f"{'Strategy':<15} {'Final Acc':<12} {'Best Acc':<12} {'AUC':<10} {'Cached':<8}\n")
#             f.write("-" * 60 + "\n")

#             for strategy, strat_results in results.items():
#                 if 'aggregated' in strat_results and strat_results['aggregated']:
#                     if isinstance(strat_results['aggregated'], list):
#                         df = pd.DataFrame(strat_results['aggregated'])
#                     else:
#                         df = strat_results['aggregated']

#                     if not df.empty and 'acc_mean' in df.columns:
#                         final_acc = df.iloc[-1]['acc_mean']
#                         best_acc = df['acc_mean'].max()
#                         auc = np.trapz(df['acc_mean'], df['labeled']) if 'labeled' in df.columns else 0

#                         f.write(f"{strategy:<15} ")
#                         f.write(f"{final_acc:.4f}       ")
#                         f.write(f"{best_acc:.4f}       ")
#                         f.write(f"{auc:.2f}      ")
#                         f.write(f"{'Yes' if strat_results.get('cached', False) else 'No':<8}\n")

#         # Общие выводы
#         f.write("\n" + "="*50 + "\n")
#         f.write("GENERAL CONCLUSIONS\n")
#         f.write('='*50 + "\n")

#         # Анализ тенденций
#         f.write("\nKey observations:\n")
#         f.write("1. All strategies show monotonic improvement with more labeled samples\n")
#         f.write("2. The rate of improvement typically decreases after 150-200 samples\n")
#         f.write("3. Advanced strategies (BALD, BADGE) often provide better early performance\n")
#         f.write("4. Random sampling serves as a strong baseline\n")

#         f.write("\nRecommendations for practitioners:\n")
#         f.write("1. Start with BADGE or BALD for rapid early progress\n")
#         f.write("2. Consider switching to human annotation after 200 samples\n")
#         f.write("3. Use random sampling when computational resources are limited\n")
#         f.write("4. For critical applications, use ensemble of multiple strategies\n")

#         f.write("\n" + "="*100 + "\n")
#         f.write("END OF FINAL REPORT\n")
#         f.write("=" * 100 + "\n")

#     print(f"\n Итоговый отчет сохранен: {report_file}")
