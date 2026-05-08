"""
sp500_feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
Единый модуль расчёта признаков — точная копия пайплайна из тетради
sp500_coursework_v6.ipynb (ячейки 17, 19, 25, 28).

Используется обоими DAG-ами:
  • dag_daily_inference.py   — инференс (ежедневно)
  • dag_annual_retrain.py    — дообучение (ежегодно)

Входные данные поступают из yfinance, выходные данные — DataFrame с 273
признаками, готовый для подачи в RandomForestClassifier.

Порядок признаков ФИКСИРОВАН и соответствует model.feature_names_in_
из random_forest_final.pkl (проверено: 273 features).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── Константы ───────────────────────────────────────────────────────────────

# Горизонты доходностей (ячейка 19 / 25 тетради)
K_VALUES: List[int] = [1, 5, 10, 15, 20, 40, 60, 80, 100, 120, 150, 180, 260]

# Окна для скользящих статистик (ячейка 28 тетради)
ROLLING_WINDOWS: List[int] = [2, 5, 10, 20]

# Базовые признаки: X_1, X_5, ..., X_260
BASE_FEAT_COLS: List[str] = [f"X_{k}" for k in K_VALUES]

# Расширенные признаки: X_{k}_{stat}_{window}
_EXT: List[str] = []
for _k in K_VALUES:
    for _w in ROLLING_WINDOWS:
        for _stat in ["mean", "std", "min", "max", "cv"]:
            _EXT.append(f"X_{_k}_{_stat}_{_w}")

ALL_FEAT_COLS: List[str] = BASE_FEAT_COLS + _EXT  # итого 273

# Минимальная история (дней), необходимая для X_260 + rolling_20 + запас
MIN_HISTORY_DAYS: int = 300

# Список тикеров, на которых обучена модель
# При необходимости замените или загружайте из внешнего источника
SP500_TICKERS: List[str] = [
    'ABBV',
 'ABT',
 'ADM',
 'AES',
 'AJG',
 'ALB',
 'ALL',
 'ALLE',
 'AMP',
 'AMZN',
 'ANET',
 'ANSS',
 'AOS',
 'APH',
 'AXON',
 'AXP',
 'BAX',
 'BBY',
 'BDX',
 'BEN',
 'BK',
 'BKR',
 'BLDR',
 'BRO',
 'BSX',
 'BWA',
 'C',
 'CAT',
 'CCL',
 'CE',
 'CLX',
 'CMCSA',
 'CME',
 'COF',
 'CRM',
 'CSCO',
 'CSGP',
 'CVX',
 'DE',
 'DELL',
 'DFS',
 'DG',
 'DHI',
 'DIS',
 'DLR',
 'DOV',
 'DPZ',
 'DTE',
 'DXCM',
 'EA',
 'ED',
 'EFX',
 'EIX',
 'EL',
 'EMN',
 'ENPH',
 'EQIX',
 'EQR',
 'EQT',
 'ESS',
 'EXPE',
 'FAST',
 'FCX',
 'FDS',
 'FDX',
 'FFIV',
 'FOX',
 'FOXA',
 'FRT',
 'FSLR',
 'FTNT',
 'GDDY',
 'GIS',
 'GLW',
 'GOOG',
 'GPN',
 'HCA',
 'HIG',
 'HII',
 'HLT',
 'HPE',
 'HRL',
 'HSIC',
 'HST',
 'ICE',
 'IEX',
 'IFF',
 'INCY',
 'INVH',
 'IP',
 'IQV',
 'IR',
 'IRM',
 'ISRG',
 'JBHT',
 'JCI',
 'K',
 'KIM',
 'KMX',
 'LEN',
 'LH',
 'LKQ',
 'LRCX',
 'MCD',
 'META',
 'MO',
 'MRK',
 'MSFT',
 'MSI',
 'MTD',
 'NDAQ',
 'NFLX',
 'NTAP',
 'NVDA',
 'O',
 'OMC',
 'ON',
 'ORCL',
 'PARA',
 'PAYX',
 'PEG',
 'PEP',
 'PFG',
 'PGR',
 'PH',
 'PHM',
 'PKG',
 'PM',
 'PNC',
 'PNW',
 'PODD',
 'PPG',
 'PPL',
 'PWR',
 'RF',
 'RJF',
 'RTX',
 'RVTY',
 'SBUX',
 'SHW',
 'SO',
 'SOLV',
 'STZ',
 'SWK',
 'SYF',
 'T',
 'TDG',
 'TJX',
 'TSLA',
 'TSN',
 'TT',
 'TTWO',
 'UAL',
 'UBER',
 'UDR',
 'UHS',
 'UPS',
 'USB',
 'V',
 'VLO',
 'VLTO',
 'VRSK',
 'VRTX',
 'VTRS',
 'WDC',
 'WEC',
 'WELL',
 'WMB',
 'WRB',
 'WST',
 'WTW',
 'XYL'
]

# Тикер индекса в yfinance
INDEX_TICKER: str = "^GSPC"


# ─── Загрузка данных из yfinance ─────────────────────────────────────────────

def fetch_stock_prices(
    tickers: List[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Загружает дневные цены закрытия для списка тикеров через yfinance.

    Возвращает DataFrame с колонками: Symbol, Date, Close
    """
    logger.info("Загрузка цен акций: %d тикеров, %s → %s", len(tickers), start, end)

    raw = yf.download(
        tickers=tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance возвращает MultiIndex columns (Price, Symbol) или просто Close
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    # Wide → Long
    df = close.stack(future_stack=True).reset_index()
    df.columns = ["Date", "Symbol", "Close"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Close"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    logger.info("Загружено строк цен: %d", len(df))
    return df


def fetch_index_prices(
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Загружает дневные цены закрытия индекса S&P 500 (^GSPC).

    Возвращает DataFrame с колонками: Date, SP500
    """
    logger.info("Загрузка индекса S&P 500: %s → %s", start, end)

    raw = yf.download(
        tickers=INDEX_TICKER,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"].copy()
        df = df.reset_index()
        df.columns = ["Date", "SP500"]
    else:
        df = raw[["Close"]].reset_index()
        df.columns = ["Date", "SP500"]

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["SP500"]).sort_values("Date").reset_index(drop=True)

    logger.info("Загружено строк индекса: %d", len(df))
    return df


# ─── Расчёт относительных доходностей (ячейки 17, 19, 25 тетради) ────────────

def _compute_index_returns(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет k-дневные доходности индекса RSP_k для всех k в K_VALUES.
    Соответствует ячейке 19 тетради.
    """
    df = index_df.sort_values("Date").reset_index(drop=True).copy()
    for k in K_VALUES:
        df[f"RSP_{k}"] = (df["SP500"] - df["SP500"].shift(k)) / df["SP500"].shift(k) * 100
    return df


def _compute_stock_features(
    price_df: pd.DataFrame,
    index_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Вычисляет базовые признаки X_k = R_stock(k) - RSP_k для одной акции.
    Соответствует ячейке 25 тетради.

    Параметры
    ----------
    price_df      : DataFrame с колонками [Symbol, Date, Close]
    index_returns : DataFrame с колонками [Date, RSP_1, RSP_5, ..., RSP_260]

    Возвращает
    ----------
    DataFrame с колонками [Symbol, Date, Close, X_1, X_5, ..., X_260]
    """
    rsp_cols = [f"RSP_{k}" for k in K_VALUES]
    merged = price_df.merge(index_returns[["Date"] + rsp_cols], on="Date", how="left")
    merged = merged.sort_values(["Symbol", "Date"])

    result_parts = []
    for sym, grp in merged.groupby("Symbol", sort=False):
        grp = grp.sort_values("Date").copy()
        for k in K_VALUES:
            r_stock = (grp["Close"] - grp["Close"].shift(k)) / grp["Close"].shift(k) * 100
            grp[f"X_{k}"] = r_stock - grp[f"RSP_{k}"]
        # Убираем вспомогательные RSP-колонки
        grp = grp.drop(columns=rsp_cols, errors="ignore")
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


def _add_statistical_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет скользящие статистики (mean, std, min, max, cv) для каждого X_k
    на окнах [2, 5, 10, 20]. Соответствует ячейке 28 тетради.

    ВАЖНО: rolling() вызывается без .shift(), как в исходном коде тетради.
    Утечки данных нет, т.к. эти признаки описывают прошлые значения X_k,
    которые сами по себе уже рассчитаны по сдвинутым ценам.
    """
    result_parts = []
    for sym, grp in daily_df.groupby("Symbol", sort=False):
        grp = grp.sort_values("Date").reset_index(drop=True).copy()
        for k in K_VALUES:
            feat = f"X_{k}"
            for w in ROLLING_WINDOWS:
                roll = grp[feat].rolling(window=w, min_periods=1)
                grp[f"{feat}_mean_{w}"] = roll.mean()
                grp[f"{feat}_std_{w}"]  = roll.std()
                grp[f"{feat}_min_{w}"]  = roll.min()
                grp[f"{feat}_max_{w}"]  = roll.max()
                # Coefficient of variation (нормированная std)
                abs_roll_mean = grp[feat].abs().rolling(window=w, min_periods=1).mean()
                grp[f"{feat}_cv_{w}"] = (
                    grp[f"{feat}_std_{w}"] / (abs_roll_mean + 1e-6)
                )
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ─── Расчёт целевой переменной (ячейка 28 тетради) ───────────────────────────

def compute_target(
    daily_df: pd.DataFrame,
    index_df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Вычисляет бинарную целевую переменную:
        y(i) = 1 если max_{k=1..horizon} [R_stock(i→i+k) - R_SP(i→i+k)] > threshold%

    Нужен только при дообучении (DAG #2).
    Для инференса (DAG #1) не используется.
    """
    sp = index_df[["Date", "SP500"]].copy()
    df = daily_df.merge(sp, on="Date", how="left")

    target_parts = []
    for sym, grp in df.groupby("Symbol", sort=False):
        grp = grp.sort_values("Date").reset_index(drop=True).copy()
        max_rel = pd.Series(-999.0, index=grp.index)

        for k in range(1, horizon + 1):
            r_stock = (grp["Close"].shift(-k) / grp["Close"] - 1) * 100
            r_sp    = (grp["SP500"].shift(-k)  / grp["SP500"]  - 1) * 100
            rel     = (r_stock - r_sp).fillna(-999)
            max_rel = np.maximum(max_rel, rel)

        grp["Target"] = (max_rel > threshold).astype(int)
        grp = grp[max_rel > -999]  # убираем строки без будущих данных
        target_parts.append(grp)

    result = pd.concat(target_parts, ignore_index=True)
    return result.drop(columns=["SP500"], errors="ignore")


# ─── Главная функция пайплайна ────────────────────────────────────────────────

def build_feature_matrix(
    tickers: List[str],
    start: date,
    end: date,
    compute_target_var: bool = False,
    index_df: Optional[pd.DataFrame] = None,
    stock_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Полный пайплайн от сырых данных до матрицы признаков.

    Параметры
    ----------
    tickers           : список тикеров для загрузки
    start             : начало периода
    end               : конец периода (exclusive, как в yfinance)
    compute_target_var: если True — вычисляет колонку Target
    index_df          : передать готовый DataFrame индекса (опционально)
    stock_df          : передать готовый DataFrame цен акций (опционально)

    Возвращает
    ----------
    (daily_df, index_df_out)
    daily_df   — DataFrame с ALL_FEAT_COLS (273 признака) + Symbol, Date, Close
    index_df_out — DataFrame индекса (для повторного использования)
    """
    # 1. Загрузка данных
    if stock_df is None:
        stock_df = fetch_stock_prices(tickers, start, end)
    if index_df is None:
        index_df = fetch_index_prices(start, end)

    # 2. Доходности индекса
    index_with_returns = _compute_index_returns(index_df)

    # 3. Базовые признаки X_k
    logger.info("Вычисляем базовые признаки X_k...")
    daily_df = _compute_stock_features(stock_df, index_with_returns)

    # 4. Расширенные признаки (rolling statistics)
    logger.info("Вычисляем расширенные признаки (rolling stats)...")
    daily_df = _add_statistical_features(daily_df)

    # 5. Целевая переменная (только для дообучения)
    if compute_target_var:
        logger.info("Вычисляем целевую переменную...")
        daily_df = compute_target(daily_df, index_df)

    # 6. Финальная фильтрация
    daily_df = daily_df.dropna(subset=ALL_FEAT_COLS)
    daily_df = daily_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # 7. Фильтрация акций с недостаточной историей
    sym_counts = daily_df.groupby("Symbol").size()
    valid_syms = sym_counts[sym_counts >= 20].index
    daily_df   = daily_df[daily_df["Symbol"].isin(valid_syms)].copy()

    logger.info(
        "Feature matrix ready: %d rows, %d symbols, %d features",
        len(daily_df),
        daily_df["Symbol"].nunique(),
        len(ALL_FEAT_COLS),
    )
    return daily_df, index_df


def get_latest_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Для инференса: оставляет только самую последнюю строку по каждому тикеру.
    Это вектор признаков «на сегодня», который подаётся в модель.
    """
    latest = (
        daily_df
        .sort_values("Date")
        .groupby("Symbol", sort=False)
        .last()
        .reset_index()
    )
    return latest[["Symbol", "Date"] + ALL_FEAT_COLS]
