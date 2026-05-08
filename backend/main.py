

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
import numpy as np

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="StockForecast API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_DIR = Path(os.getenv("PREDICTIONS_DIR", Path(__file__).parent / "predictions"))
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_REPORT_FILE = Path(__file__).parent / "backtest_full_report.json"


# ─── Schemas ─────────────────────────────────────────────────────────────────

class TopPick(BaseModel):
    rank: int
    symbol: str
    prob_growth: float
    signal_date: str

class Prediction(BaseModel):
    symbol: str
    prob_growth: float

class PredictionMetadata(BaseModel):
    prediction_date: str
    model: str
    model_version: str
    total_stocks_scored: int
    top_k: int
    target: str
    generated_at: str

class PredictionPayload(BaseModel):
    metadata: PredictionMetadata
    top_picks: List[TopPick]
    all_predictions: List[Prediction]


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_backtest_report() -> Dict[str, Any]:
    """Возвращает отчёт по бэктесту"""
    if not BACKTEST_REPORT_FILE.exists():
        raise HTTPException(status_code=404, detail="Backtest report not available")
    with open(BACKTEST_REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ─── Predictions ────────────────────────────────────────────────────────────

@app.post("/api/predictions")
def receive_predictions(payload: PredictionPayload):
    date_str = payload.metadata.prediction_date
    data = payload.model_dump()

    dated_path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
    with open(dated_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    latest_path = PREDICTIONS_DIR / "predictions_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "inserted": len(payload.all_predictions), "date": date_str}


@app.get("/api/predictions/latest")
def get_latest() -> Dict[str, Any]:
    latest_path = PREDICTIONS_DIR / "predictions_latest.json"
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail="No predictions yet")
    with open(latest_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/predictions/history")
def get_history() -> List[str]:
    files = sorted(PREDICTIONS_DIR.glob("predictions_2*.json"), reverse=True)
    return [f.stem.replace("predictions_", "") for f in files]


# ─── Backtest ───────────────────────────────────────────────────────────────

@app.get("/api/backtest/report")
def get_backtest_report_endpoint() -> Dict[str, Any]:
    return get_backtest_report()


@app.get("/api/backtest/summary")
def get_backtest_summary() -> Dict[str, Any]:
    report = get_backtest_report()
    if "summary" in report:
        return report["summary"]
    return {}


@app.get("/api/backtest/equity")
def get_backtest_equity() -> List[Dict[str, Any]]:
    report = get_backtest_report()
    equity = report.get("equity_curve", [])
    trades = report.get("trades", [])
    
    # Добавляем даты из trades к equity curve
    result = []
    for i, e in enumerate(equity):
        item = e.copy()
        if i < len(trades):
            item['Date'] = trades[i].get('Date', f'Day {i+1}')
        else:
            item['Date'] = f'Day {i+1}'
        result.append(item)
    
    return result


@app.get("/api/backtest/trades")
def get_backtest_trades() -> List[Dict[str, Any]]:
    report = get_backtest_report()
    return report.get("trades", [])


@app.get("/api/backtest/charts/equity")
def get_equity_chart_data() -> Dict[str, Any]:
    """Возвращает данные для графика кривой капитала"""
    try:
        report = get_backtest_report()
        equity = report.get("equity_curve", [])
        trades = report.get("trades", [])
        
        if not equity:
            return {"dates": [], "equity": [], "drawdown": [], "pnl": []}
        
        # Получаем даты из trades
        dates = []
        for i, e in enumerate(equity):
            if i < len(trades):
                dates.append(trades[i].get('Date', f'Day {i+1}'))
            else:
                dates.append(f'Day {i+1}')
        
        equity_values = [e.get('Capital', 0) for e in equity]
        pnl_values = [e.get('PnL', 0) for e in equity]
        
        # Рассчитываем просадку
        current_max = equity_values[0] if equity_values else 0
        drawdowns = []
        for value in equity_values:
            if value > current_max:
                current_max = value
            dd = (current_max - value) / current_max if current_max > 0 else 0
            drawdowns.append(dd)
        
        return {
            "dates": dates,
            "equity": equity_values,
            "drawdown": drawdowns,
            "pnl": pnl_values
        }
    except Exception as e:
        return {"dates": [], "equity": [], "drawdown": [], "pnl": [], "error": str(e)}


@app.get("/api/backtest/charts/distribution")
def get_distribution_chart_data() -> Dict[str, Any]:
    """Возвращает данные для распределения P&L"""
    try:
        report = get_backtest_report()
        trades = report.get("trades", [])
        
        if not trades:
            return {"pnl_distribution": [], "win_loss": {"wins": 0, "losses": 0}}
        
        pnl_values = [t.get('Net PnL ($)', 0) for t in trades]
        wins = sum(1 for p in pnl_values if p > 0)
        losses = sum(1 for p in pnl_values if p <= 0)
        
        return {
            "pnl_distribution": pnl_values,
            "win_loss": {"wins": wins, "losses": losses}
        }
    except Exception as e:
        return {"pnl_distribution": [], "win_loss": {"wins": 0, "losses": 0}, "error": str(e)}


@app.get("/api/backtest/charts/top_tickers")
def get_top_tickers_chart_data() -> Dict[str, Any]:
    """Возвращает данные для графика топ тикеров"""
    try:
        report = get_backtest_report()
        trades = report.get("trades", [])
        
        if not trades:
            return {"symbols": [], "counts": [], "avg_returns": []}
        
        symbol_counts = Counter([t.get('Symbol', '') for t in trades])
        top_symbols = symbol_counts.most_common(10)
        
        symbol_returns = {}
        for t in trades:
            symbol = t.get('Symbol', '')
            ret = t.get('Forward Return (10d)', 0)
            if symbol not in symbol_returns:
                symbol_returns[symbol] = []
            symbol_returns[symbol].append(ret)
        
        avg_returns = [float(np.mean(symbol_returns[s])) for s, _ in top_symbols]
        
        return {
            "symbols": [s for s, _ in top_symbols],
            "counts": [c for _, c in top_symbols],
            "avg_returns": avg_returns
        }
    except Exception as e:
        return {"symbols": [], "counts": [], "avg_returns": [], "error": str(e)}


@app.post("/api/backtest/run")
def run_backtest_endpoint() -> Dict[str, Any]:
    """Запускает бэктест заново"""
    import subprocess
    import sys
    
    try:
        result = subprocess.run(
            [sys.executable, "backtest_full.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return get_backtest_report()
        else:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {result.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)