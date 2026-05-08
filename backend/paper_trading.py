"""
Paper Trading Simulator
───────────────────────
Симуляция торговли на основе прогнозов модели.

Логика:
- Каждый торговый день (когда поступают новые predictions) покупаем топ-N акций
- Держим позиции K дней, затем продаём
- Перебалансировка: новые покупки только на свободные средства
- Фиксируем все сделки и историю капитала
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Конфигурация симуляции
PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"
INITIAL_CAPITAL = 10_000.0  # Начальный капитал в долларах
TOP_K_TO_BUY = 5            # Сколько топ-акций покупаем
HOLDING_DAYS = 10           # Сколько дней держим позицию
MAX_POSITIONS = 10          # Максимум открытых позиций


def _empty_portfolio() -> Dict[str, Any]:
    """Возвращает пустую структуру портфеля"""
    return {
        "capital": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "positions": [],           # Открытые позиции
        "closed_trades": [],       # Закрытые сделки
        "equity_curve": [          # История капитала
            {"date": datetime.now().date().isoformat(), "value": INITIAL_CAPITAL}
        ],
        "last_processed_date": None,  # Дата последнего обработанного прогноза
    }


def load_portfolio() -> Dict[str, Any]:
    """Загружает портфель из файла"""
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            logger.error("Ошибка чтения portfolio.json, создаём новый")
            return _empty_portfolio()
    return _empty_portfolio()


def save_portfolio(portfolio: Dict[str, Any]) -> None:
    """Сохраняет портфель в файл"""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def get_portfolio_stats(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """Возвращает статистику портфеля"""
    total_capital = portfolio["cash"]
    total_pnl = 0
    open_positions_value = 0
    
    # Считаем стоимость открытых позиций
    for pos in portfolio.get("positions", []):
        current_price = pos.get("current_price", pos["entry_price"])
        position_value = pos["quantity"] * current_price
        open_positions_value += position_value
        total_pnl += (current_price - pos["entry_price"]) * pos["quantity"]
    
    total_capital += open_positions_value
    
    # Статистика по закрытым сделкам
    closed_trades = portfolio.get("closed_trades", [])
    winning_trades = [t for t in closed_trades if t.get("profit", 0) > 0]
    losing_trades = [t for t in closed_trades if t.get("profit", 0) <= 0]
    
    total_profit = sum(t.get("profit", 0) for t in closed_trades)
    total_return_percent = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
    
    return {
        "total_capital": round(total_capital, 2),
        "cash_balance": round(portfolio["cash"], 2),
        "open_positions_value": round(open_positions_value, 2),
        "total_profit": round(total_profit, 2),
        "total_return_percent": round(total_return_percent, 2),
        "win_rate": round(win_rate, 4),
        "total_trades": len(closed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "open_positions": len(portfolio.get("positions", [])),
    }


def update_current_prices(portfolio: Dict[str, Any], predictions: Dict[str, float]) -> None:
    """
    Обновляет текущие цены для открытых позиций
    predictions: {symbol: prob_growth} — используем вероятности как прокси для цены
    В реальности нужно подставлять реальные цены из stock_df
    """
    for pos in portfolio.get("positions", []):
        symbol = pos["symbol"]
        # Используем вероятность как суррогат цены (в реальности брать Close из stock_df)
        prob = predictions.get(symbol, pos.get("current_price", pos["entry_price"]))
        pos["current_price"] = prob
        pos["pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["quantity"]


def close_expired_positions(portfolio: Dict[str, Any], current_date: str) -> None:
    """
    Закрывает позиции, которые держатся больше HOLDING_DAYS дней
    """
    positions = portfolio.get("positions", [])
    to_close = []
    
    for i, pos in enumerate(positions):
        open_date = datetime.fromisoformat(pos["open_date"])
        current_dt = datetime.fromisoformat(current_date)
        days_held = (current_dt - open_date).days
        
        if days_held >= HOLDING_DAYS:
            to_close.append(i)
    
    # Закрываем позиции с конца, чтобы индексы не сбивались
    for i in reversed(to_close):
        pos = positions[i]
        exit_price = pos.get("current_price", pos["entry_price"])
        profit = (exit_price - pos["entry_price"]) * pos["quantity"]
        
        # Добавляем в закрытые сделки
        portfolio["closed_trades"].append({
            "symbol": pos["symbol"],
            "open_date": pos["open_date"],
            "close_date": current_date,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "quantity": pos["quantity"],
            "profit": round(profit, 2),
            "type": "sell",
        })
        
        # Возвращаем средства в cash
        portfolio["cash"] += pos["quantity"] * exit_price
        
        # Удаляем позицию
        positions.pop(i)
    
    portfolio["positions"] = positions


def open_new_positions(
    portfolio: Dict[str, Any], 
    top_picks: List[Dict[str, Any]], 
    current_date: str
) -> None:
    """
    Открывает новые позиции на основе топ-пиков
    """
    cash_available = portfolio["cash"]
    positions_count = len(portfolio.get("positions", []))
    
    # Сколько ещё можно открыть позиций
    slots_available = MAX_POSITIONS - positions_count
    
    if slots_available <= 0 or cash_available <= 0:
        return
    
    # Берем топ-N акций, которых ещё нет в портфеле
    current_symbols = {p["symbol"] for p in portfolio.get("positions", [])}
    new_picks = [
        pick for pick in top_picks 
        if pick["symbol"] not in current_symbols
    ][:min(slots_available, TOP_K_TO_BUY)]
    
    if not new_picks:
        return
    
    # Распределяем cash поровну между новыми позициями
    amount_per_position = cash_available / len(new_picks)
    
    for pick in new_picks:
        symbol = pick["symbol"]
        prob = pick["prob_growth"]
        
        # Цена входа — используем вероятность как прокси
        # В реальности брать Close из stock_df
        entry_price = prob * 100  # Нормализуем до цены 0-100
        
        # Количество акций = сумма / цена
        quantity = amount_per_position / entry_price
        
        portfolio["positions"].append({
            "symbol": symbol,
            "open_date": current_date,
            "entry_price": round(entry_price, 4),
            "quantity": round(quantity, 4),
            "current_price": entry_price,
            "pnl": 0,
            "prob_at_entry": prob,
        })
        
        # Списываем cash
        portfolio["cash"] -= amount_per_position
        
        # Добавляем запись о покупке в закрытые сделки (как историю)
        portfolio["closed_trades"].append({
            "symbol": symbol,
            "open_date": current_date,
            "close_date": None,
            "entry_price": round(entry_price, 4),
            "exit_price": None,
            "quantity": round(quantity, 4),
            "profit": None,
            "type": "buy",
        })
    
    logger.info(f"Открыто {len(new_picks)} новых позиций: {[p['symbol'] for p in new_picks]}")


def update_equity_curve(portfolio: Dict[str, Any], current_date: str) -> None:
    """
    Обновляет кривую капитала
    """
    stats = get_portfolio_stats(portfolio)
    equity_curve = portfolio.get("equity_curve", [])
    
    # Проверяем, не было ли уже записи за этот день
    if equity_curve and equity_curve[-1]["date"] == current_date:
        equity_curve[-1]["value"] = stats["total_capital"]
    else:
        equity_curve.append({
            "date": current_date,
            "value": stats["total_capital"]
        })
    
    portfolio["equity_curve"] = equity_curve


def process_new_predictions(predictions_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Основная функция: обрабатывает новый прогноз и обновляет портфель
    
    Вызывается каждый раз, когда Airflow присылает новые predictions
    """
    portfolio = load_portfolio()
    
    metadata = predictions_data.get("metadata", {})
    prediction_date = metadata.get("prediction_date")
    
    if not prediction_date:
        logger.error("Нет prediction_date в данных")
        return portfolio
    
    # Проверяем, не обрабатывали ли уже этот день
    if portfolio.get("last_processed_date") == prediction_date:
        logger.info(f"Прогноз на {prediction_date} уже обработан, пропускаем")
        return portfolio
    
    # Создаём словарь цен/вероятностей для всех акций
    all_predictions = {
        p["symbol"]: p["prob_growth"] 
        for p in predictions_data.get("all_predictions", [])
    }
    
    # Обновляем текущие цены для открытых позиций
    update_current_prices(portfolio, all_predictions)
    
    # Закрываем позиции, которые держались HOLDING_DAYS
    close_expired_positions(portfolio, prediction_date)
    
    # Открываем новые позиции
    top_picks = predictions_data.get("top_picks", [])
    open_new_positions(portfolio, top_picks, prediction_date)
    
    # Обновляем кривую капитала
    update_equity_curve(portfolio, prediction_date)
    
    # Сохраняем дату последней обработки
    portfolio["last_processed_date"] = prediction_date
    
    # Сохраняем портфель
    save_portfolio(portfolio)
    
    # Логируем результат
    stats = get_portfolio_stats(portfolio)
    logger.info(
        f"Portfolio updated for {prediction_date}: "
        f"Capital: ${stats['total_capital']:.2f}, "
        f"Cash: ${stats['cash_balance']:.2f}, "
        f"Open positions: {stats['open_positions']}, "
        f"Total trades: {stats['total_trades']}"
    )
    
    return portfolio


def reset_portfolio() -> None:
    """Сбрасывает портфель до начального состояния"""
    save_portfolio(_empty_portfolio())
    logger.info("Portfolio reset to initial state")


# Если запускаем файл напрямую — тестируем
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Создаём тестовый прогноз
    test_predictions = {
        "metadata": {"prediction_date": "2026-05-09"},
        "top_picks": [
            {"rank": 1, "symbol": "AAPL", "prob_growth": 0.95, "signal_date": "2026-05-09"},
            {"rank": 2, "symbol": "MSFT", "prob_growth": 0.92, "signal_date": "2026-05-09"},
            {"rank": 3, "symbol": "GOOGL", "prob_growth": 0.89, "signal_date": "2026-05-09"},
            {"rank": 4, "symbol": "AMZN", "prob_growth": 0.87, "signal_date": "2026-05-09"},
            {"rank": 5, "symbol": "META", "prob_growth": 0.85, "signal_date": "2026-05-09"},
        ],
        "all_predictions": [
            {"symbol": "AAPL", "prob_growth": 0.95},
            {"symbol": "MSFT", "prob_growth": 0.92},
            {"symbol": "GOOGL", "prob_growth": 0.89},
            {"symbol": "AMZN", "prob_growth": 0.87},
            {"symbol": "META", "prob_growth": 0.85},
        ]
    }
    
    print("Testing paper trading...")
    result = process_new_predictions(test_predictions)
    print(f"Result: {result['cash']=}, {len(result['positions'])=}")
