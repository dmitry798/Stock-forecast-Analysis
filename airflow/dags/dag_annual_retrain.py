"""
dag_annual_retrain.py
─────────────────────────────────────────────────────────────────────────────
DAG #2 — Ежегодное дообучение модели

Расписание: 1 января каждого года в 03:00 UTC (или ручной триггер)

Граф задач:
  fetch_new_stock_data ──► fetch_new_index_data
          │                        │
          └──────────┬─────────────┘
                     ▼
             build_features_with_target
                     │
                     ▼
        sliding_window_retrain          ← точный алгоритм из тетради (ячейка 90)
                     │
                     ▼
            evaluate_new_model
                     │
                     ▼
            compare_with_baseline
                     │
                 ┌───┴───┐
                 ▼       ▼
        promote_model   reject_model

Параметры скользящего окна (из тетради):
  TRAIN_DAYS = 253  (~1 торговый год)
  GAP_DAYS   = 10   (embargo — предотвращает утечку через таргет)
  TEST_DAYS  = 21   (~1 торговый месяц)
  STEP_DAYS  = 21   (шаг сдвига)

Конфигурация (Variables Airflow):
  SP500_MODEL_PATH          — путь к текущей production-модели
  SP500_MODEL_BACKUP_DIR    — папка для архива старых моделей
  SP500_RETRAIN_START_DATE  — начало периода дообучения (default: 2025-01-01)
  SP500_MIN_AUC_DELTA       — минимальный прирост ROC-AUC для замены модели (default: -0.005)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago

import sys
sys.path.insert(0, os.path.dirname(__file__))
from sp500_feature_engineering import (
    SP500_TICKERS,
    ALL_FEAT_COLS,
    build_feature_matrix,
    fetch_index_prices,
    fetch_stock_prices,
)

logger = logging.getLogger(__name__)

# ─── Гиперпараметры модели (из тетради, ячейка 90) ───────────────────────────

RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth":    8,
    "criterion":    "entropy",
    "n_jobs":       -1,
    "random_state": 42,
}

# Параметры скользящего окна (из тетради, ячейка 90)
TRAIN_DAYS: int = 253   # ~1 торговый год
GAP_DAYS:   int = 10    # embargo — предотвращает утечку через горизонт таргета
TEST_DAYS:  int = 21    # ~1 торговый месяц
STEP_DAYS:  int = 21    # шаг сдвига окна

DEFAULT_ARGS = {
    "owner":             "ml-team",
    "depends_on_past":   False,
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=30),
}


# ─── DAG ─────────────────────────────────────────────────────────────────────

@dag(
    dag_id="sp500_annual_retrain",
    description="Ежегодное дообучение RandomForest скользящим окном",
    schedule_interval="0 3 1 1 *",  # 1 января в 03:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["sp500", "retrain", "annual"],
    doc_md=__doc__,
    # Разрешаем ручной запуск с конфигурацией:
    # airflow dags trigger sp500_annual_retrain --conf '{"retrain_start":"2025-01-01"}'
    params={
        "retrain_start": "2025-01-01",
        "force_promote":  False,   # True — заменить модель даже при отрицательном delta AUC
    },
)
def sp500_annual_retrain():

    # ── Task 1: Загрузка новых данных акций ───────────────────────────────────
    @task
    def fetch_new_stock_data(**context) -> str:
        """
        Загружает данные с retrain_start по сегодняшний день.
        retrain_start задаётся через DAG params или Variable.
        """
        params         = context.get("params", {})
        retrain_start  = params.get(
            "retrain_start",
            Variable.get("SP500_RETRAIN_START_DATE", default_var="2025-01-01"),
        )
        start_dt = date.fromisoformat(retrain_start)
        end_dt   = date.today() + timedelta(days=1)

        logger.info("Загрузка данных акций: %s → %s", start_dt, end_dt)
        df = fetch_stock_prices(SP500_TICKERS, start_dt, end_dt)

        tmp_dir  = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path   = str(tmp_dir / f"retrain_stocks_{run_ts}.parquet")

        df.to_parquet(path, index=False)
        logger.info("Цены акций сохранены: %s (%d строк)", path, len(df))
        return path

    # ── Task 2: Загрузка данных индекса ──────────────────────────────────────
    @task
    def fetch_new_index_data(**context) -> str:
        """Загружает данные индекса за тот же период."""
        params        = context.get("params", {})
        retrain_start = params.get(
            "retrain_start",
            Variable.get("SP500_RETRAIN_START_DATE", default_var="2025-01-01"),
        )
        start_dt = date.fromisoformat(retrain_start)
        end_dt   = date.today() + timedelta(days=1)

        df = fetch_index_prices(start_dt, end_dt)

        tmp_dir = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path    = str(tmp_dir / f"retrain_index_{run_ts}.parquet")

        df.to_parquet(path, index=False)
        logger.info("Индекс сохранён: %s (%d строк)", path, len(df))
        return path

    # ── Task 3: Построение признаков + целевой переменной ────────────────────
    @task
    def build_features_with_target(stock_path: str, index_path: str) -> str:
        """
        Строит матрицу признаков (273 фичи) + целевую переменную Target.
        Соответствует ячейкам 25, 28, 29 тетради.
        """
        stock_df = pd.read_parquet(stock_path)
        index_df = pd.read_parquet(index_path)

        daily_df, _ = build_feature_matrix(
            tickers=stock_df["Symbol"].unique().tolist(),
            start=stock_df["Date"].min().date(),
            end=(stock_df["Date"].max().date() + timedelta(days=1)),
            compute_target_var=True,  # ← нужна для обучения
            stock_df=stock_df,
            index_df=index_df,
        )

        # Убираем строки без таргета
        daily_df = daily_df.dropna(subset=ALL_FEAT_COLS + ["Target"])

        tmp_dir = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path    = str(tmp_dir / f"retrain_features_{run_ts}.parquet")

        daily_df.to_parquet(path, index=False)
        logger.info(
            "Feature matrix с таргетом: %d строк, %d тикеров, p(1)=%.3f",
            len(daily_df),
            daily_df["Symbol"].nunique(),
            daily_df["Target"].mean(),
        )
        return path

    # ── Task 4: Дообучение скользящим окном (точная копия ячейки 90) ─────────
    @task
    def sliding_window_retrain(features_path: str) -> str:
        """
        Реализует алгоритм скользящего окна из тетради (ячейка 90).

        Параметры окна:
          TRAIN_DAYS = 253 (~1 год)
          GAP_DAYS   = 10  (embargo)
          TEST_DAYS  = 21  (~1 месяц)
          STEP_DAYS  = 21  (шаг)

        На последнем фолде обученная модель сохраняется как финальная.
        Метрики каждого фолда накапливаются и возвращаются вместе с pkl.
        """
        daily_df = pd.read_parquet(features_path)

        # Полностью воспроизводим ячейку 90 тетради
        df_sorted_sw = daily_df.sort_values("Date").reset_index(drop=True)
        df_sorted_sw = df_sorted_sw.dropna(subset=ALL_FEAT_COLS + ["Target"])

        unique_dates = np.sort(df_sorted_sw["Date"].unique())
        n_dates      = len(unique_dates)

        logger.info(
            "Скользящее окно: %d торговых дней | %s → %s",
            n_dates,
            pd.to_datetime(unique_dates[0]).date(),
            pd.to_datetime(unique_dates[-1]).date(),
        )
        logger.info(
            "Параметры: TRAIN=%d, GAP=%d, TEST=%d, STEP=%d",
            TRAIN_DAYS, GAP_DAYS, TEST_DAYS, STEP_DAYS,
        )

        fold_scores: List[float] = []
        last_model: Optional[RandomForestClassifier] = None
        date_cursor = 0
        fold_num    = 0

        while True:
            train_start_idx = date_cursor
            train_end_idx   = date_cursor + TRAIN_DAYS
            test_start_idx  = train_end_idx + GAP_DAYS
            test_end_idx    = test_start_idx + TEST_DAYS

            if test_end_idx > n_dates:
                break

            train_dates = unique_dates[train_start_idx:train_end_idx]
            test_dates  = unique_dates[test_start_idx:test_end_idx]

            train_fold = df_sorted_sw[df_sorted_sw["Date"].isin(train_dates)]
            test_fold  = df_sorted_sw[df_sorted_sw["Date"].isin(test_dates)]

            X_tr = train_fold[ALL_FEAT_COLS]
            y_tr = train_fold["Target"]
            X_te = test_fold[ALL_FEAT_COLS]
            y_te = test_fold["Target"]

            if len(y_tr) < 50 or len(y_te) < 10:
                logger.debug("Фолд пропущен (мало данных): train=%d, test=%d", len(y_tr), len(y_te))
                date_cursor += STEP_DAYS
                continue

            # Обучаем модель с теми же гиперпараметрами, что в тетради
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_tr, y_tr)

            y_prob = model.predict_proba(X_te)[:, 1]

            # Защита: если в тесте только один класс — пропускаем метрику
            if len(np.unique(y_te)) < 2:
                logger.warning(
                    "Фолд %d: только один класс в тесте, ROC-AUC не определён.",
                    fold_num + 1,
                )
                date_cursor += STEP_DAYS
                continue

            auc = roc_auc_score(y_te, y_prob)
            fold_scores.append(auc)
            last_model = model
            fold_num  += 1

            logger.info(
                "Фолд %2d | train: %s→%s (%d obs) | test: %s→%s (%d obs) | AUC=%.4f",
                fold_num,
                pd.to_datetime(train_dates[0]).date(),
                pd.to_datetime(train_dates[-1]).date(),
                len(y_tr),
                pd.to_datetime(test_dates[0]).date(),
                pd.to_datetime(test_dates[-1]).date(),
                len(y_te),
                auc,
            )

            date_cursor += STEP_DAYS

        if last_model is None or fold_num == 0:
            raise RuntimeError(
                "Скользящее окно не создало ни одного фолда. "
                f"Проверьте период данных (дней в датасете: {n_dates})."
            )

        mean_auc = float(np.mean(fold_scores))
        std_auc  = float(np.std(fold_scores))
        logger.info(
            "Скользящее окно завершено: %d фолдов | ROC-AUC: %.4f ± %.4f",
            fold_num, mean_auc, std_auc,
        )

        # Сохраняем модель и метрики
        tmp_dir = Path(Variable.get("SP500_TMP_DIR", default_var="/tmp/sp500"))
        run_ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_path   = str(tmp_dir / f"new_model_{run_ts}.pkl")
        metrics_path = str(tmp_dir / f"new_model_metrics_{run_ts}.json")

        joblib.dump(last_model, model_path)

        metrics = {
            "n_folds":         fold_num,
            "mean_auc":        mean_auc,
            "std_auc":         std_auc,
            "fold_aucs":       fold_scores,
            "train_days":      TRAIN_DAYS,
            "gap_days":        GAP_DAYS,
            "test_days":       TEST_DAYS,
            "step_days":       STEP_DAYS,
            "rf_params":       RF_PARAMS,
            "retrain_date":    date.today().isoformat(),
            "n_features":      len(ALL_FEAT_COLS),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Новая модель сохранена: %s", model_path)
        logger.info("Метрики сохранены: %s", metrics_path)

        # Возвращаем пути через XCom
        return json.dumps({"model_path": model_path, "metrics_path": metrics_path})

    # ── Task 5: Оценка новой модели на hold-out ───────────────────────────────
    @task
    def evaluate_new_model(
        retrain_result_json: str,
        features_path: str,
    ) -> str:
        """
        Оценивает новую модель на hold-out выборке (последние 20% по дате).
        Сравнивает с текущей production-моделью.
        """
        retrain_result = json.loads(retrain_result_json)
        new_model_path = retrain_result["model_path"]
        metrics_path   = retrain_result["metrics_path"]

        daily_df = pd.read_parquet(features_path)
        daily_df = daily_df.sort_values("Date").reset_index(drop=True)
        daily_df = daily_df.dropna(subset=ALL_FEAT_COLS + ["Target"])

        # Hold-out: последние 20% по дате
        unique_dates = daily_df["Date"].unique()
        cutoff_idx   = int(len(unique_dates) * 0.8)
        cutoff_date  = unique_dates[cutoff_idx]

        holdout = daily_df[daily_df["Date"] >= cutoff_date]
        X_hold  = holdout[ALL_FEAT_COLS]
        y_hold  = holdout["Target"]

        if len(np.unique(y_hold)) < 2:
            logger.warning("Hold-out: только один класс, метрики пропущены.")
            eval_metrics = {"holdout_auc": None, "holdout_ap": None}
        else:
            new_model = joblib.load(new_model_path)
            y_prob    = new_model.predict_proba(X_hold)[:, 1]

            eval_metrics = {
                "holdout_auc":    float(roc_auc_score(y_hold, y_prob).round(4)),
                "holdout_ap":     float(average_precision_score(y_hold, y_prob).round(4)),
                "holdout_period": [
                    str(holdout["Date"].min())[:10],
                    str(holdout["Date"].max())[:10],
                ],
                "holdout_size":   len(holdout),
                "p1_holdout":     float(y_hold.mean().round(4)),
            }
            logger.info(
                "Hold-out оценка новой модели: AUC=%.4f, AP=%.4f",
                eval_metrics["holdout_auc"],
                eval_metrics["holdout_ap"],
            )

        # Оцениваем текущую production-модель на том же hold-out
        prod_model_path = Variable.get(
            "SP500_MODEL_PATH",
            default_var="/opt/airflow/models/random_forest_final.pkl",
        )
        try:
            prod_model  = joblib.load(prod_model_path)
            y_prob_prod = prod_model.predict_proba(X_hold)[:, 1]
            eval_metrics["baseline_auc"] = float(
                roc_auc_score(y_hold, y_prob_prod).round(4)
                if len(np.unique(y_hold)) >= 2 else 0.5
            )
            eval_metrics["baseline_ap"] = float(
                average_precision_score(y_hold, y_prob_prod).round(4)
                if len(np.unique(y_hold)) >= 2 else float(y_hold.mean().round(4))
            )
            logger.info(
                "Hold-out оценка baseline: AUC=%.4f, AP=%.4f",
                eval_metrics["baseline_auc"],
                eval_metrics["baseline_ap"],
            )
        except FileNotFoundError:
            logger.warning("Production-модель не найдена: %s", prod_model_path)
            eval_metrics["baseline_auc"] = None
            eval_metrics["baseline_ap"]  = None

        # Обновляем файл метрик
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
        all_metrics.update(eval_metrics)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info("Метрики обновлены: %s", metrics_path)
        return json.dumps({
            "model_path":   new_model_path,
            "metrics_path": metrics_path,
            "eval_metrics": eval_metrics,
        })

    # ── Task 6: Сравнение с baseline и принятие решения ──────────────────────
    @task
    def compare_with_baseline(eval_result_json: str, **context) -> str:
        """
        Сравнивает новую модель с production-baseline.
        Решает: promote / reject.

        Критерий промоушена:
          new_holdout_auc >= baseline_auc - MIN_AUC_DELTA

        MIN_AUC_DELTA по умолчанию = -0.005 (допускаем потерю до 0.5%)
        Если задан params.force_promote=True — всегда продвигаем.
        """
        eval_result = json.loads(eval_result_json)
        eval_metrics = eval_result["eval_metrics"]
        params       = context.get("params", {})

        force_promote  = params.get("force_promote", False)
        min_auc_delta  = float(Variable.get("SP500_MIN_AUC_DELTA", default_var="-0.005"))

        new_auc      = eval_metrics.get("holdout_auc")
        baseline_auc = eval_metrics.get("baseline_auc")

        if force_promote:
            decision = "promote"
            reason   = "force_promote=True задан в params"
        elif new_auc is None:
            decision = "reject"
            reason   = "holdout_auc не вычислен (только один класс в hold-out)"
        elif baseline_auc is None:
            decision = "promote"
            reason   = "Production-модель не найдена, промоутируем новую по умолчанию"
        else:
            delta = new_auc - baseline_auc
            if delta >= min_auc_delta:
                decision = "promote"
                reason   = f"AUC delta={delta:+.4f} ≥ порог {min_auc_delta}"
            else:
                decision = "reject"
                reason   = f"AUC delta={delta:+.4f} < порог {min_auc_delta}"

        logger.info("Решение: %s | %s", decision.upper(), reason)
        logger.info(
            "  Новая модель: AUC=%.4f | Baseline: %s",
            new_auc or 0,
            f"{baseline_auc:.4f}" if baseline_auc else "N/A",
        )

        result = {
            "model_path":   eval_result["model_path"],
            "metrics_path": eval_result["metrics_path"],
            "decision":     decision,
            "reason":       reason,
            "new_auc":      new_auc,
            "baseline_auc": baseline_auc,
        }
        return json.dumps(result)

    # ── Task 7: Продвижение модели (promote) ─────────────────────────────────
    @task
    def promote_model(compare_result_json: str) -> None:
        """
        Если решение = promote:
          1. Копирует текущую production-модель в backup
          2. Заменяет pkl новой моделью
          3. Обновляет Variable SP500_MODEL_VERSION
          4. Сохраняет отчёт о дообучении
        """
        result = json.loads(compare_result_json)

        if result["decision"] != "promote":
            logger.info(
                "Решение = reject, продвижение пропущено. Причина: %s",
                result["reason"],
            )
            return

        prod_model_path = Variable.get(
            "SP500_MODEL_PATH",
            default_var="/opt/airflow/models/random_forest_final.pkl",
        )
        backup_dir = Path(
            Variable.get("SP500_MODEL_BACKUP_DIR", default_var="/opt/airflow/models/backup")
        )
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Бэкап текущей модели
        if Path(prod_model_path).exists():
            ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup = backup_dir / f"random_forest_{ts}.pkl"
            shutil.copy2(prod_model_path, backup)
            logger.info("Текущая модель сохранена в backup: %s", backup)

        # Замена модели
        Path(prod_model_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result["model_path"], prod_model_path)
        logger.info("Новая модель промоутирована: %s", prod_model_path)

        # Обновление версии
        new_version = datetime.utcnow().strftime("v%Y%m%d")
        Variable.set("SP500_MODEL_VERSION", new_version)
        logger.info("Версия модели обновлена: %s", new_version)

        # Сохранение отчёта о дообучении
        metrics_path = result["metrics_path"]
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["promoted"]        = True
        metrics["promoted_at"]     = datetime.utcnow().isoformat() + "Z"
        metrics["new_version"]     = new_version
        metrics["decision_reason"] = result["reason"]

        report_dir = Path(
            Variable.get("SP500_PREDICTIONS_DIR", default_var="/opt/airflow/predictions")
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"retrain_report_{new_version}.json"

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Отчёт о дообучении сохранён: %s", report_path)
        logger.info(
            "=== ДООБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===\n"
            "  Новая версия:   %s\n"
            "  AUC (hold-out): %.4f (было: %s)\n"
            "  Причина:        %s",
            new_version,
            result["new_auc"] or 0,
            f"{result['baseline_auc']:.4f}" if result["baseline_auc"] else "N/A",
            result["reason"],
        )

    # ── Task 8: Отклонение модели (reject) ───────────────────────────────────
    @task
    def reject_model(compare_result_json: str) -> None:
        """
        Если решение = reject: логирует отклонение, сохраняет отчёт.
        Новый pkl остаётся в /tmp для ручного ревью.
        """
        result = json.loads(compare_result_json)

        if result["decision"] == "promote":
            logger.info("Модель промоутирована, задача reject_model не выполняется.")
            return

        logger.warning(
            "МОДЕЛЬ ОТКЛОНЕНА. Причина: %s\n"
            "  Новая AUC:    %.4f\n"
            "  Baseline AUC: %s\n"
            "  Модель доступна для ручного ревью: %s",
            result["reason"],
            result["new_auc"] or 0,
            f"{result['baseline_auc']:.4f}" if result["baseline_auc"] else "N/A",
            result["model_path"],
        )

        # Сохраняем отчёт об отклонении
        metrics_path = result["metrics_path"]
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["promoted"]        = False
        metrics["rejected_at"]     = datetime.utcnow().isoformat() + "Z"
        metrics["decision_reason"] = result["reason"]

        report_dir = Path(
            Variable.get("SP500_PREDICTIONS_DIR", default_var="/opt/airflow/predictions")
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"retrain_REJECTED_{ts}.json"

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Отчёт об отклонении: %s", report_path)

    # ─── Граф задач ──────────────────────────────────────────────────────────
    stock_path   = fetch_new_stock_data()
    index_path   = fetch_new_index_data()
    feat_path    = build_features_with_target(stock_path, index_path)
    retrain_json = sliding_window_retrain(feat_path)
    eval_json    = evaluate_new_model(retrain_json, feat_path)
    compare_json = compare_with_baseline(eval_json)

    promote_model(compare_json)
    reject_model(compare_json)


# Регистрация DAG
dag_instance = sp500_annual_retrain()
