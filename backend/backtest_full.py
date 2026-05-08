"""
Бэктест стратегии прогнозирования роста акций S&P 500
Период: 2026-04-01 → 2026-05-08
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройки
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10

# Конфигурация
PREDICTIONS_DIR = Path(__file__).parent / "predictions"
INITIAL_CAPITAL = 10_000
TOP_K = 10
REBALANCE_DAYS = 5
COMMISSION = 0.001

START_DATE = "2026-04-01"
END_DATE = "2026-05-08"


def load_historical_predictions():
    """Загружает прогнозы модели"""
    predictions = {}
    for file_path in sorted(PREDICTIONS_DIR.glob("predictions_2026-*.json")):
        date_str = file_path.stem.replace("predictions_", "")
        if date_str < START_DATE or date_str > END_DATE:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions[date_str] = {
            'date': pd.to_datetime(date_str),
            'top_picks': data.get('top_picks', [])[:TOP_K],
            'all_probs': {p['symbol']: p['prob_growth'] for p in data.get('all_predictions', [])}
        }
    
    print(f"📊 Загружено прогнозов: {len(predictions)}")
    return predictions


def load_price_data():
    """Загружает цены акций"""
    stocks_path = Path("/tmp/sp500") / "stocks_20260508.pkl"
    if not stocks_path.exists():
        raise FileNotFoundError(f"Файл {stocks_path} не найден")
    
    df = pd.read_pickle(stocks_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].astype(float)
    
    price_pivot = df.pivot(index='Date', columns='Symbol', values='Close')
    
    print(f"📈 Загружено цен: {len(df):,} строк, {df['Symbol'].nunique()} активов")
    return df, price_pivot


def calculate_forward_returns(price_pivot, dates, forward_days=10):
    """Рассчитывает форвардную доходность"""
    forward_returns = {}
    
    for symbol in price_pivot.columns:
        forward_returns[symbol] = {}
        for date in dates:
            if date in price_pivot.index:
                current_price = price_pivot.loc[date, symbol] if symbol in price_pivot.columns else np.nan
                
                date_idx = price_pivot.index.get_loc(date) if date in price_pivot.index else None
                if date_idx is not None and date_idx + forward_days < len(price_pivot.index):
                    future_date = price_pivot.index[date_idx + forward_days]
                    future_price = price_pivot.loc[future_date, symbol] if symbol in price_pivot.columns else np.nan
                    
                    if not np.isnan(current_price) and not np.isnan(future_price) and current_price > 0:
                        forward_returns[symbol][date] = (future_price / current_price) - 1
    
    return forward_returns


def run_backtest():
    """Запускает бэктест"""
    print("\n" + "="*70)
    print("🚀 ЗАПУСК БЭКТЕСТА")
    print("="*70)
    print(f"Период: {START_DATE} → {END_DATE}")
    print(f"Стратегия: топ-{TOP_K} акций, ребалансировка каждые {REBALANCE_DAYS} дней")
    print(f"Комиссия: {COMMISSION*100}%\n")
    
    # Загружаем данные
    predictions = load_historical_predictions()
    df_prices, price_pivot = load_price_data()
    
    # Получаем все торговые даты
    all_dates = sorted(pd.date_range(start=START_DATE, end=END_DATE, freq='B'))
    all_dates = [d for d in all_dates if d in price_pivot.index]
    
    # Рассчитываем форвардную доходность
    forward_returns = calculate_forward_returns(price_pivot, all_dates, forward_days=10)
    
    # Выбираем даты для ребалансировки
    rebalance_dates = all_dates[::REBALANCE_DAYS]
    print(f"📅 Всего торговых дней: {len(all_dates)}")
    print(f"🔄 Ребалансировок: {len(rebalance_dates)}\n")
    
    # Инициализация
    capital = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    
    # Проходим по каждой дате ребалансировки
    for i, entry_date in enumerate(rebalance_dates):
        date_str = entry_date.strftime('%Y-%m-%d')
        
        # Проверяем, есть ли прогноз на эту дату
        if date_str not in predictions:
            print(f"⚠️  Нет прогноза на {date_str}, пропускаем")
            equity_curve.append({'Date': entry_date, 'Capital': capital})
            continue
        
        # Получаем топ-акции из прогноза
        top_picks = predictions[date_str]['top_picks']
        top_symbols = [p['symbol'] for p in top_picks[:TOP_K]]
        top_probs = {p['symbol']: p['prob_growth'] for p in top_picks[:TOP_K]}
        
        # Рассчитываем размер позиции
        position_size = capital / TOP_K
        
        period_pnl = 0
        period_trades = []
        
        # Проходим по каждой выбранной акции
        for symbol in top_symbols:
            # Проверяем, есть ли цена на входе
            if entry_date not in price_pivot.index:
                continue
            
            entry_price = price_pivot.loc[entry_date, symbol] if symbol in price_pivot.columns else np.nan
            
            if np.isnan(entry_price) or entry_price <= 0:
                continue
            
            # Получаем форвардную доходность
            fwd_return = forward_returns.get(symbol, {}).get(entry_date, 0)
            
            # Рассчитываем P&L
            gross_pnl = position_size * fwd_return
            commission_cost = position_size * COMMISSION * 2
            net_pnl = gross_pnl - commission_cost
            
            period_pnl += net_pnl
            
            period_trades.append({
                'Date': entry_date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Probability': top_probs[symbol],
                'Entry Price': entry_price,
                'Forward Return (10d)': fwd_return,
                'Gross PnL ($)': gross_pnl,
                'Commission ($)': commission_cost,
                'Net PnL ($)': net_pnl,
            })
            
            trades.append(period_trades[-1])
        
        # Обновляем капитал
        capital += period_pnl
        
        # Сохраняем кривую капитала
        equity_curve.append({
            'Date': entry_date,
            'Capital': capital,
            'PnL': period_pnl,
            'Trades': len(period_trades)
        })
        
        print(f"📅 {date_str}: Capital: ${capital:,.2f} | PnL: ${period_pnl:+.2f} | Позиций: {len(period_trades)}")
    
    # Создаем DataFrame
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trades_df = pd.DataFrame(trades)
    
    # Рассчитываем метрики
    final_capital = equity_df['Capital'].iloc[-1] if len(equity_df) > 0 else INITIAL_CAPITAL
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Ежедневные доходности для Sharpe
    daily_returns = equity_df['Capital'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Максимальная просадка
    rolling_max = equity_df['Capital'].cummax()
    drawdown = (equity_df['Capital'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['Net PnL ($)'] > 0]
        losing_trades = trades_df[trades_df['Net PnL ($)'] <= 0]
        win_rate = len(winning_trades) / len(trades_df)
        total_gross_profit = winning_trades['Net PnL ($)'].sum() if len(winning_trades) > 0 else 0
        total_gross_loss = abs(losing_trades['Net PnL ($)'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
    
    # Buy-and-Hold
    bh_return = 0
    bh_capital = INITIAL_CAPITAL
    if START_DATE in price_pivot.index.strftime('%Y-%m-%d').tolist() and END_DATE in price_pivot.index.strftime('%Y-%m-%d').tolist():
        bh_returns = []
        for symbol in price_pivot.columns:
            start_price = price_pivot.loc[START_DATE, symbol] if symbol in price_pivot.columns else np.nan
            end_price = price_pivot.loc[END_DATE, symbol] if symbol in price_pivot.columns else np.nan
            if not np.isnan(start_price) and not np.isnan(end_price) and start_price > 0:
                bh_returns.append((end_price / start_price) - 1)
        bh_return = np.mean(bh_returns) if bh_returns else 0
        bh_capital = INITIAL_CAPITAL * (1 + bh_return)
    
    alpha = total_return - bh_return
    
    # Вывод результатов
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*70)
    print(f"Период:               {START_DATE} → {END_DATE}")
    print(f"Стартовый капитал:    ${INITIAL_CAPITAL:>10,.0f}")
    print(f"Финальный капитал:    ${final_capital:>10,.2f}")
    print("─"*70)
    print(f"Суммарная доходность: {total_return:>+10.2%}")
    print(f"Sharpe Ratio:         {sharpe_ratio:>10.2f}")
    print(f"Макс. просадка:       {max_drawdown:>10.2%}")
    print("─"*70)
    print(f"Всего сделок:         {len(trades_df):>10,}")
    print(f"Win rate:             {win_rate:>10.1%}")
    print(f"Profit Factor:        {profit_factor:>10.2f}")
    print("─"*70)
    print(f"Buy-and-Hold:         {bh_return:>+10.2%}  (${bh_capital:,.0f})")
    print(f"Alpha vs B&H:         {alpha:>+10.2%}")
    print("="*70)
    
    # Сохраняем результаты
    output_path = Path(__file__).parent / "backtest_full_report.json"
    report = {
        'equity_curve': equity_df.to_dict(orient='records'),
        'trades': trades_df.to_dict(orient='records') if len(trades_df) > 0 else [],
        'summary': {
            'final_capital': float(final_capital),
            'total_return_pct': float(total_return * 100),
            'total_trades': len(trades_df),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'alpha': float(alpha)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Результаты сохранены в {output_path}")
    
    return equity_df, trades_df


if __name__ == "__main__":
    print("\n" + "🎯"*35)
    print("БЭКТЕСТ СТРАТЕГИИ ПРОГНОЗИРОВАНИЯ S&P 500")
    print("🎯"*35)
    
    equity_df, trades_df = run_backtest()
    print("\n✅ Бэктест завершён!")
