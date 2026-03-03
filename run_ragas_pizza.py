"""
Автотесты LLM-ответов для пиццерии с метриками RAGAS + fallback для answer_relevancy.

1) Генерирует ответы моделью OpenAI по базе знаний пиццерии (меню, доставка, акции).
2) Считает метрики RAGAS (faithfulness, context_precision, context_recall).
3) Если включён fallback (по умолчанию), считает answer_relevancy как cos_sim(emb(Q), emb(A)) в [0..1].

ENV:
- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-4o-mini)
- RAGAS_EMBEDDING_MODEL (default: text-embedding-3-small)
- USE_SIMPLE_AR = "1" (default) — включить fallback answer_relevancy; "0" — попытаться RAGAS AnswerRelevancy
- THRESH_* — пороги метрик
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from openai import OpenAI

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings  # для остальных метрик
from langchain_openai import ChatOpenAI
from datasets import Dataset

load_dotenv()

YC_API_KEY   = (os.getenv("YC_API_KEY") or "").strip()
YC_FOLDER_ID = (os.getenv("YC_FOLDER_ID") or "").strip()

if not YC_API_KEY or not YC_FOLDER_ID:
    raise RuntimeError("Нужны YC_API_KEY и YC_FOLDER_ID в .env")

OPENAI_MODEL = (
    os.getenv("OPENAI_MODEL")
    or f"gpt://{YC_FOLDER_ID}/yandexgpt-lite/latest"
).strip()

RAGAS_EMBEDDING_MODEL = (
    os.getenv("RAGAS_EMBEDDING_MODEL")
    or f"emb://{YC_FOLDER_ID}/text-embeddings/latest"
).strip()

client = OpenAI(
    api_key="DUMMY",
    base_url="https://llm.api.cloud.yandex.net/v1",
    default_headers={
        "Authorization": f"Api-Key {YC_API_KEY}",
        "OpenAI-Project": YC_FOLDER_ID,
    },
)

USE_SIMPLE_AR = os.getenv("USE_SIMPLE_AR", "1").strip() not in ("0", "false", "False")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ragas_demo_pizzeria")

# ---------------- Данные для пиццерии ----------------

SYSTEM_RULES = (
    "Вы — ассистент службы поддержки пиццерии «У Лунтика». Отвечайте ТОЛЬКО фактами из контекста. "
    "Если ответа нет в контексте — скажите: «Извините, не нашёл такой информации в базе знаний пиццерии»."
)

DOCS = {
    "menu": "Меню: Пепперони (550₽), Маргарита (450₽), Четыре сыра (600₽), Гавайская (520₽), Мясная (650₽). Напитки: Кола, Спрайт, Фанта — по 100₽.",
    "delivery": "Доставка: бесплатно при заказе от 800₽. Время доставки: 30-60 минут в пределах города. Работаем с 10:00 до 23:00.",
    "payment": "Оплата: наличными курьеру, картой онлайн или картой курьеру. Принимаем Visa, Mastercard, МИР.",
    "promos": "Акции: «3 пиццы по цене 2» каждый вторник. Скидка 20% на первый заказ через приложение. Подарок при заказе от 1500₽.",
    "allergens": "Аллергены: Пепперони содержит глютен, молочные продукты. Маргарита — лактоза. Уточняйте состав у оператора.",
    "work_hours": "Часы работы: ежедневно с 10:00 до 23:00. Доставка работает по тому же графику.",
    "contacts": "Контакты: +7 (999) 123-45-67, адрес: ул. Пиццерийная, д. 1. Мы в Instagram: @luntik_pizza",
}

@dataclass
class Sample:
    question: str
    ground_truth: str
    contexts: List[str]

SAMPLES: List[Sample] = [
    Sample(
        question="Сколько стоит пепперони и есть ли доставка?",
        ground_truth="Пепперони — 550₽. Доставка бесплатна от 800₽, время доставки 30-60 минут.",
        contexts=[DOCS["menu"], DOCS["delivery"]],
    ),
    Sample(
        question="Какие акции у вас сейчас действуют?",
        ground_truth="Акции: «3 пиццы по цене 2» по вторникам, скидка 20% на первый заказ, подарок от 1500₽.",
        contexts=[DOCS["promos"]],
    ),
    Sample(
        question="До скольки работает доставка?",
        ground_truth="Доставка работает ежедневно с 10:00 до 23:00.",
        contexts=[DOCS["work_hours"], DOCS["delivery"]],
    ),
    Sample(
        question="Какими картами можно оплатить?",
        ground_truth="Принимаем Visa, Mastercard, МИР. Можно оплатить наличными или картой курьеру, а также онлайн.",
        contexts=[DOCS["payment"]],
    ),
    Sample(
        question="Есть ли в меню пицца без лактозы?",
        ground_truth="Извините, не нашёл такой информации в базе знаний пиццерии. Пожалуйста, уточните у оператора.",
        contexts=[DOCS["allergens"]],  # В контексте нет безлактозной пиццы
    ),
    Sample(
        question="Какой адрес пиццерии и телефон?",
        ground_truth="Адрес: ул. Пиццерийная, д. 1. Телефон: +7 (999) 123-45-67. Мы в Instagram: @luntik_pizza",
        contexts=[DOCS["contacts"]],
    ),
    Sample(
        question="Что входит в состав пиццы Четыре сыра?",
        ground_truth="Пицца «Четыре сыра» стоит 600₽. Точный состав уточняйте у оператора.",
        contexts=[DOCS["menu"]],  # В меню только цена, состава нет
    ),
    Sample(
        question="Какая столица Франции?",
        ground_truth="Париж.",
        contexts=[], # QA-режим (без контекста) — проверка, что модель не уходит в пиццерийную тему
    ),
]

# Функции для LLM, эмбеддингов и метрик
def extract_output_text(resp) -> str:
    """Извлекает текст из Responses API (output_text или output[].content[].text)."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    try:
        parts = []
        for block in getattr(resp, "output", []):
            for c in getattr(block, "content", []):
                if getattr(c, "type", None) in ("output_text", "text") and getattr(c, "text", None):
                    parts.append(c.text)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass
    
    return str(resp)

# Генерация ответа
def llm_answer(question: str, contexts: List[str]) -> str:
    """Генерация ответа через OpenAI Responses API (температура=0).
    Если contexts пуст — модель отвечает из своих знаний (QA-режим)."""
    if contexts and len(contexts) > 0:
        joined_ctx = "\n---\n".join(contexts)
        prompt = (
            f"{SYSTEM_RULES}\n\n"
            f"Контекст (база знаний пиццерии):\n{joined_ctx}\n\n"
            f"Вопрос клиента: {question}\nКраткий ответ:"
        )
    else:
        # QA-режим: без контекста и без правила «только из контекста»
        prompt = (
            "Вы — фактологичный помощник. Отвечайте кратко и точно.\n\n"
            f"Вопрос: {question}\nКраткий ответ:"
        )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_RULES if contexts else "Вы — фактологичный помощник. Отвечайте кратко и точно."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
    )

    return (resp.choices[0].message.content or "").strip()


# Косинусное сходство
def cosine(u: List[float], v: List[float]) -> float:
    """Косинусное сходство -> [0,1]."""
    import math
    s = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return max(0.0, min(1.0, (s / (nu * nv) + 1.0) / 2.0))

# Эмбеддинги
def embed_texts(texts: List[str], *, model: str, client: OpenAI) -> List[List[float]]:
    vecs: List[List[float]] = []
    for t in texts:
        res = client.embeddings.create(
            model=model,
            input=str(t),
            encoding_format="float",
        )
        vecs.append(res.data[0].embedding)
    return vecs

# Метрики
def compute_simple_answer_relevancy_from_df(df_texts: pd.DataFrame, *, model: str, client: OpenAI) -> List[float]:
    """Surrogate для answer_relevancy: cos_sim(emb(Q), emb(A)) из ИСХОДНОГО df с колонками question/answer."""
    questions = df_texts["question"].astype(str).tolist()
    answers = df_texts["answer"].astype(str).tolist()
    q_vecs = embed_texts(questions, model=model, client=client)
    a_vecs = embed_texts(answers, model=model, client=client)
    return [cosine(q, a) for q, a in zip(q_vecs, a_vecs)]

# Метрики
def compute_answer_gt_similarity(df_texts: pd.DataFrame, *, model: str, client: OpenAI) -> List[float]:
    """Семантическая корректность ответа: cos_sim(emb(A), emb(GT)) в [0..1]."""
    answers = df_texts["answer"].astype(str).tolist()
    gts = df_texts["ground_truth"].astype(str).tolist()
    a_vecs = embed_texts(answers, model=model, client=client)
    g_vecs = embed_texts(gts, model=model, client=client)
    return [cosine(a, g) for a, g in zip(a_vecs, g_vecs)]


# Основной сценарий тестирования
def main() -> None:
    print("\n🍕 АВТОТЕСТЫ АССИСТЕНТА ПИЦЦЕРИИ «У ЛУНТИКА» 🍕")
    print("="*60)
    
    # Генерация ответов
    rows: List[Dict[str, Any]] = []
    print("\n[1/3] Генерация ответов моделью...")
    log.info("Начало генерации: %d кейсов, модель=%s", len(SAMPLES), OPENAI_MODEL)

    for s in tqdm(SAMPLES, desc="Генерация ответов"):
        answer = llm_answer(s.question, s.contexts)
        rows.append(
            {
                "question": s.question,
                "answer": answer,
                "ground_truth": s.ground_truth,
                "contexts": list(s.contexts),
            }
        )

    df = pd.DataFrame(rows)

    # Базовая валидация
    bad = []
    for i, r in df.iterrows():
        if not (isinstance(r["question"], str) and r["question"].strip()):
            bad.append((i, "question"))
        if not (isinstance(r["answer"], str) and r["answer"].strip()):
            bad.append((i, "answer"))
        if not (isinstance(r["ground_truth"], str) and r["ground_truth"].strip()):
            bad.append((i, "ground_truth"))
        # contexts может быть пустым списком в QA-режиме
        if not (isinstance(r["contexts"], list) and all(isinstance(c, str) for c in r["contexts"])):
            bad.append((i, "contexts"))


      # Оценка: RAGAS для кейсов с контекстом + универсальные QA-метрики
    print("\n[2/3] Оценка метрик...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {YC_API_KEY}",
                "OpenAI-Project": YC_FOLDER_ID,
            },
        )
    )

    evaluator_embeddings = OpenAIEmbeddings(client=client, model=RAGAS_EMBEDDING_MODEL)

    rag_mask = df["contexts"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
    details_all = pd.DataFrame(index=df.index)


    # RAGAS (только там, где есть контекст)
   
    if rag_mask.any():
        hf_ds = Dataset.from_pandas(df.loc[rag_mask, ["question", "answer", "contexts", "ground_truth"]])
        metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]
        if not USE_SIMPLE_AR:
            from ragas.metrics import AnswerRelevancy
            metrics.insert(1, AnswerRelevancy())

        log.info(
            "ragas.evaluate(): rows=%d, metrics=%s, emb_model=%s, USE_SIMPLE_AR=%s",
            len(hf_ds),
            [m.__class__.__name__ for m in metrics],
            RAGAS_EMBEDDING_MODEL,
            USE_SIMPLE_AR,
        )
        try:
            result = evaluate(
                dataset=hf_ds,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                show_progress=True,
                raise_exceptions=True,
                batch_size=4,
            )
            try:
                details_rag = result.to_pandas()
            except AttributeError:
                details_rag = pd.DataFrame(result)
            # синхронизируем индексы с исходным df
            details_rag.index = df.index[rag_mask]
            # вливаем только доступные колонки
            for col in details_rag.columns:
                details_all.loc[rag_mask, col] = details_rag[col]
        except Exception as e:
            print("\n[FAIL] RAGAS evaluate() завершился ошибкой:", type(e).__name__, str(e))
            sys.exit(1)

    # AnswerRelevancy (fallback Q↔A) для всех строк
    ar_scores = compute_simple_answer_relevancy_from_df(df, model=RAGAS_EMBEDDING_MODEL, client=client)
    details_all["answer_relevancy"] = ar_scores

    # Семантическая корректность (A↔GT) для всех строк
    qa_sim = compute_answer_gt_similarity(df, model=RAGAS_EMBEDDING_MODEL, client=client)
    details_all["qa_semantic_correctness"] = qa_sim

    # Сводка и quality-gates
    print("\n[3/3] Итоги (средние значения метрик для пиццерии):")

    wanted = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "qa_semantic_correctness",   
    ]
    present = [c for c in wanted if c in details_all.columns]
    summary: Dict[str, float] = {c: float(details_all[c].mean(skipna=True)) for c in present}

    for k in wanted:
        v = summary.get(k, None)
        print(f"- {k:23}: " + ("N/A" if v is None or (v != v) else f"{v:.4f}"))

    log.info("Итоговые метрики: %s", summary)

    def env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return default

    thresholds = {
        "faithfulness": env_float("THRESH_FAITHFULNESS", 0.80),
        "answer_relevancy": env_float("THRESH_ANSWER_RELEVANCY", 0.50),
        "context_precision": env_float("THRESH_CONTEXT_PRECISION", 0.50),
        "context_recall": env_float("THRESH_CONTEXT_RECALL", 0.70),
        "qa_semantic_correctness": env_float("THRESH_QA_SIM", 0.80),  
    }

    # Подробный отчёт по кейсам
    metrics_df = None
    if "details_all" in locals():
        metrics_df = details_all.copy()
    else:
        metrics_df = pd.DataFrame(index=df.index)

    report = df.join(metrics_df, how="left")

    def _fmt(x):
        try:
            import math
            if x is None or (isinstance(x, float) and (math.isnan(x))):
                return "N/A"
            return f"{float(x):.4f}"
        except Exception:
            return "N/A"

    print("\n=== ПОДРОБНЫЙ ОТЧЁТ ПО КЕЙСАМ ===")
    for i, r in report.iterrows():
        print(f"\n[{i+1}] Вопрос: {r['question']}")
        print(f"     Ответ: {r['answer']}")
        print(f"    Эталон: {r['ground_truth']}")
        parts = []
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "qa_semantic_correctness"]:
            if m in report.columns:
                parts.append(f"{m}={_fmt(r.get(m))}")
        print("Метрики: " + (", ".join(parts) if parts else "нет доступных метрик"))


    failed = []
    for k, th in thresholds.items():
        if k not in present:
            continue  
        v = summary.get(k, None)
        if v is None or (v != v) or v < th:
            failed.append(k)

    if failed:
        print("\n[FAIL] Порог(и) не пройдены:", failed)
        print("Нужно дообучить модель или дополнить базу знаний пиццерии!")
        sys.exit(1)
    else:
        print("\n[OK] Все пороги пройдены. Ассистент пиццерии работает отлично! 🍕✨")


if __name__ == "__main__":
    main()