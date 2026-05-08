"""
dag_daily_inference.py
─────────────────────────────────────────────────────────────────────────────
DAG #1 — Ежедневный инференс

Расписание: каждый торговый день в 21:00 UTC (после закрытия NYSE 16:00 ET)

Граф задач:
  check_trading_day
      │
      ▼
  fetch_stock_data ──► fetch_index_data
      │                      │
      └──────────┬───────────┘
                 ▼
         build_features
                 │
                 ▼
         run_inference
                 │
          ┌──────┴──────┐
          ▼             ▼
   save_predictions  notify_frontend   (push_to_api)
          │
          ▼
   log_run_metrics

Конфигурация (Variables Airflow):
  SP500_MODEL_PATH        — путь к pkl-файлу модели
  SP500_PREDICTIONS_DIR   — папка для сохранения JSON-результатов
  SP500_API_ENDPOINT      — URL фронтенда (опционально)
  SP500_API_TOKEN         — Bearer-токен для API (опционально)
  SP500_TOP_K             — сколько акций возвращать (default: 20)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import requests

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Shared feature engineering module (лежит рядом с DAG-ом)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sp500_feature_engineering import (
    SP500_TICKERS,
    ALL_FEAT_COLS,
    MIN_HISTORY_DAYS,
    build_feature_matrix,
    fetch_index_prices,
    fetch_stock_prices,
    get_latest_features,
)

logger = logging.getLogger(__name__)

# ─── Конфигурация по умолчанию ───────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

# NYSE выходные (список можно расширить)
_NYSE_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # Новый год
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

_NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}

NYSE_HOLIDAYS = _NYSE_HOLIDAYS_2025 | _NYSE_HOLIDAYS_2026


def _is_trading_day(d: date) -> bool:
    """True, если d — торговый день NYSE."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS  # 0–4: пн–пт


# ─── DAG ─────────────────────────────────────────────────────────────────────

@dag(
    dag_id="sp500_daily_inference",
    description="Ежедневный инференс: загрузка данных → признаки → прогноз → фронт",
    schedule_interval="0 21 * * 1-5",   # пн–пт, 21:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["sp500", "inference", "daily"],
    doc_md=__doc__,
)
def sp500_daily_inference():

    # ── Task 1: Проверка торгового дня ───────────────────────────────────────
    @task
    def check_trading_day(**context) -> bool:
        """
        Проверяет, является ли сегодня торговым днём NYSE.
        Если нет — поднимает AirflowSkipException.
        """
        from airflow.exceptions import AirflowSkipException

        exec_date = context["execution_date"].date()
        if not _is_trading_day(exec_date):
            raise AirflowSkipException(
                f"{exec_date} не является торговым днём NYSE, пропускаем DAG."
            )
        logger.info("Торговый день подтверждён: %s", exec_date)
        return True

    # ── Task 2: Загрузка цен акций ────────────────────────────────────────────
    @task
    def fetch_stock_data(**context) -> str:
        """
        Загружает MIN_HISTORY_DAYS дней цен акций через yfinance.
        Сохраняет Parquet во временную папку, возвращает путь к файлу.
        """
        end_dt   = context["execution_date"].date() + timedelta(days=1)
        start_dt = end_dt - timedelta(days=MIN_HISTORY_DAYS + 60)  # запас

        tickers = SP500_TICKERS
        df = fetch_stock_prices(tickers, start_dt, end_dt)

        tmp_dir = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_date = context["execution_date"].strftime("%Y%m%d")
        path = str(tmp_dir / f"stocks_{run_date}.parquet")

        df.to_parquet(path, index=False)
        logger.info("Цены акций сохранены: %s (%d строк)", path, len(df))
        return path

    # ── Task 3: Загрузка индекса ──────────────────────────────────────────────
    @task
    def fetch_index_data(**context) -> str:
        """
        Загружает MIN_HISTORY_DAYS дней значений индекса S&P 500.
        Сохраняет Parquet, возвращает путь.
        """
        end_dt   = context["execution_date"].date() + timedelta(days=1)
        start_dt = end_dt - timedelta(days=MIN_HISTORY_DAYS + 60)

        df = fetch_index_prices(start_dt, end_dt)

        tmp_dir  = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_date = context["execution_date"].strftime("%Y%m%d")
        path = str(tmp_dir / f"index_{run_date}.parquet")

        df.to_parquet(path, index=False)
        logger.info("Индекс сохранён: %s (%d строк)", path, len(df))
        return path

    # ── Task 4: Построение матрицы признаков ─────────────────────────────────
    @task
    def build_features(stock_path: str, index_path: str, **context) -> str:
        """
        Вычисляет все 273 признака по тетради (ячейки 25 + 28).
        Оставляет только последнюю строку по каждому тикеру (для инференса).
        Сохраняет Parquet, возвращает путь.
        """
        stock_df = pd.read_parquet(stock_path)
        index_df = pd.read_parquet(index_path)

        daily_df, _ = build_feature_matrix(
            tickers=stock_df["Symbol"].unique().tolist(),
            start=stock_df["Date"].min().date(),
            end=(stock_df["Date"].max().date() + timedelta(days=1)),
            compute_target_var=False,  # инференс: таргет не нужен
            stock_df=stock_df,
            index_df=index_df,
        )

        # Берём только последнюю строку по каждому тикеру
        latest = get_latest_features(daily_df)

        tmp_dir  = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_date = context["execution_date"].strftime("%Y%m%d")
        path = str(tmp_dir / f"features_{run_date}.parquet")

        latest.to_parquet(path, index=False)
        logger.info(
            "Матрица признаков готова: %s (%d тикеров, %d признаков)",
            path, len(latest), len(ALL_FEAT_COLS),
        )
        return path

    # ── Task 5: Инференс ──────────────────────────────────────────────────────
    @task
    def run_inference(features_path: str, **context) -> str:
        """
        Загружает RandomForestClassifier из pkl.
        Предсказывает вероятность роста каждой акции > 2% за 10 дней.
        Возвращает топ-K акций с наибольшей вероятностью.
        """
        model_path = Variable.get(
            "SP500_MODEL_PATH",
            default_var="/opt/airflow/models/random_forest_final.pkl",
        )
        top_k = int(Variable.get("SP500_TOP_K", default_var="20"))

        logger.info("Загрузка модели: %s", model_path)
        model = joblib.load(model_path)

        features_df = pd.read_parquet(features_path)

        # Проверяем, что все 273 признака присутствуют
        missing = set(ALL_FEAT_COLS) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"Отсутствуют признаки в матрице: {sorted(missing)[:10]}..."
            )

        X = features_df[ALL_FEAT_COLS]
        valid_mask = X.notna().all(axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) == 0:
            raise RuntimeError("Нет валидных строк после удаления NaN.")

        y_prob = model.predict_proba(X_valid)[:, 1]

        results = features_df[valid_mask][["Symbol", "Date"]].copy()
        results["prob_growth"] = np.round(y_prob, 6)
        results["prediction_date"] = context["execution_date"].date().isoformat()
        results = results.sort_values("prob_growth", ascending=False)

        # Полный список + топ-K
        top_results = results.head(top_k).copy()
        top_results["rank"] = range(1, len(top_results) + 1)

        tmp_dir  = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_date = context["execution_date"].strftime("%Y%m%d")
        path = str(tmp_dir / f"predictions_{run_date}.parquet")

        results.to_parquet(path, index=False)

        logger.info(
            "Инференс завершён: %d тикеров оценено, топ-%d:\n%s",
            len(results),
            top_k,
            top_results[["rank", "Symbol", "prob_growth"]].to_string(index=False),
        )
        return path

    # ── Task 6: Сохранение предсказаний (JSON) ────────────────────────────────
    @task
    def save_predictions(predictions_path: str, **context) -> Dict[str, Any]:
        """
        Сохраняет результаты инференса в JSON-файл в постоянную директорию.
        Структура JSON описана ниже.
        """
        preds_df = pd.read_parquet(predictions_path)
        top_k    = int(Variable.get("SP500_TOP_K", default_var="20"))

        output_dir = Path(
            Variable.get("SP500_PREDICTIONS_DIR", default_var="/opt/airflow/predictions")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        run_date = context["execution_date"].strftime("%Y-%m-%d")
        out_path = output_dir / f"predictions_{run_date}.json"

        # Форматирование для фронтенда
        payload: Dict[str, Any] = {
            "metadata": {
                "prediction_date": run_date,
                "model": "RandomForestClassifier",
                "model_version": Variable.get("SP500_MODEL_VERSION", default_var="1.0"),
                "total_stocks_scored": len(preds_df),
                "top_k": top_k,
                "target": "P(рост акции > рост индекса S&P500 на 2% за 10 дней)",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
            "top_picks": [
                {
                    "rank":          int(i + 1),
                    "symbol":        row["Symbol"],
                    "prob_growth":   float(row["prob_growth"]),
                    "signal_date":   str(row["Date"])[:10],
                }
                for i, row in preds_df.head(top_k).iterrows()
            ],
            "all_predictions": [
                {
                    "symbol":      row["Symbol"],
                    "prob_growth": float(row["prob_growth"]),
                }
                for _, row in preds_df.iterrows()
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Предсказания сохранены: %s", out_path)

        # Также сохраняем «latest» — постоянно обновляемый файл
        latest_path = output_dir / "predictions_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload

    # ── Task 7: Push на фронтенд (опционально) ───────────────────────────────
    @task
    def notify_frontend(payload: Dict[str, Any]) -> None:
        """
        Отправляет предсказания на REST API фронтенда (POST).
        Если SP500_API_ENDPOINT не задан — пропускает шаг.

        ── Интеграция с фронтендом ──────────────────────────────────────────
        Фронтенд должен реализовать endpoint:
          POST /api/v1/predictions
          Authorization: Bearer <token>
          Content-Type: application/json

        Тело запроса — JSON структуры payload (см. task save_predictions).

        Пример ответа (200 OK):
          {"status": "ok", "inserted": 170}

        Для автоматической торговли (брокер-интеграция):
          Рекомендуется отдельный микросервис, подписанный на топ-пики.
          Он принимает POST /api/v1/trade_signals и формирует ордера через
          Interactive Brokers API / Alpaca API.
        ─────────────────────────────────────────────────────────────────────
        """
        api_endpoint = Variable.get("SP500_API_ENDPOINT", default_var="")
        api_token    = Variable.get("SP500_API_TOKEN",    default_var="")

        if not api_endpoint:
            logger.info("SP500_API_ENDPOINT не задан, пропускаем push на фронт.")
            return

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        try:
            resp = requests.post(
                api_endpoint,
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            logger.info(
                "Данные отправлены на фронтенд: HTTP %d, тело: %s",
                resp.status_code,
                resp.text[:200],
            )
        except requests.RequestException as exc:
            # Не роняем DAG из-за ошибки доставки — только логируем
            logger.error("Ошибка при отправке на фронтенд: %s", exc)

    # ── Task 8: Метрики запуска ───────────────────────────────────────────────
    @task
    def log_run_metrics(predictions_path: str, **context) -> None:
        """
        Логирует агрегированные метрики запуска в Airflow XCom и логи.
        Может быть расширен для записи в MLflow / Prometheus.
        """
        preds = pd.read_parquet(predictions_path)
        run_date = context["execution_date"].strftime("%Y-%m-%d")

        metrics = {
            "run_date":           run_date,
            "stocks_scored":      len(preds),
            "mean_prob":          float(preds["prob_growth"].mean().round(4)),
            "median_prob":        float(preds["prob_growth"].median().round(4)),
            "pct_above_50":       float((preds["prob_growth"] > 0.5).mean().round(4)),
            "top1_symbol":        preds.iloc[0]["Symbol"],
            "top1_prob":          float(preds.iloc[0]["prob_growth"]),
            "top5_symbols":       preds.head(5)["Symbol"].tolist(),
        }

        logger.info("=== RUN METRICS [%s] ===", run_date)
        for k, v in metrics.items():
            logger.info("  %-20s : %s", k, v)

        # Сохраняем в XCom для мониторинга
        context["ti"].xcom_push(key="run_metrics", value=metrics)

    # ─── Граф задач ──────────────────────────────────────────────────────────
    trading_ok   = check_trading_day()
    stock_path   = fetch_stock_data()
    index_path   = fetch_index_data()

    stock_path.set_upstream(trading_ok)
    index_path.set_upstream(trading_ok)

    feat_path    = build_features(stock_path, index_path)
    pred_path    = run_inference(feat_path)
    payload      = save_predictions(pred_path)

    notify_frontend(payload)
    log_run_metrics(pred_path)


# Регистрация DAG
dag_instance = sp500_daily_inference()
