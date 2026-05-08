"""
Бэктест стратегии прогнозирования роста акций S&P 500
Период: 2026-04-01 → 2026-05-08
Стратегия: ребалансировка каждые 5 дней (т.к. период короткий)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройки
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Конфигурация
PREDICTIONS_DIR = Path(__file__).parent / "predictions"
INITIAL_CAPITAL = 10_000
TOP_K = 10
REBALANCE_DAYS = 5  # Ребалансировка каждые 5 дней (т.к. период короткий)
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
    
    # Создаем pivot таблицу для быстрого доступа к ценам
    price_pivot = df.pivot(index='Date', columns='Symbol', values='Close')
    
    print(f"📈 Загружено цен: {len(df):,} строк, {df['Symbol'].nunique()} активов")
    print(f"   Период: {df['Date'].min().date()} → {df['Date'].max().date()}")
    
    return df, price_pivot


def calculate_forward_returns(price_pivot, dates, forward_days=10):
    """Рассчитывает форвардную доходность для каждой даты"""
    forward_returns = {}
    
    for symbol in price_pivot.columns:
        forward_returns[symbol] = {}
        for date in dates:
            if date in price_pivot.index:
                current_price = price_pivot.loc[date, symbol] if symbol in price_pivot.columns else np.nan
                
                # Находим цену через forward_days дней
                date_idx = price_pivot.index.get_loc(date) if date in price_pivot.index else None
                if date_idx is not None and date_idx + forward_days < len(price_pivot.index):
                    future_date = price_pivot.index[date_idx + forward_days]
                    future_price = price_pivot.loc[future_date, symbol] if symbol in price_pivot.columns else np.nan
                    
                    if not np.isnan(current_price) and not np.isnan(future_price):
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
    positions_history = []
    
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
            
            if np.isnan(entry_price):
                continue
            
            # Получаем форвардную доходность
            fwd_return = forward_returns.get(symbol, {}).get(entry_date, 0)
            
            # Рассчитываем P&L
            gross_pnl = position_size * fwd_return
            commission_cost = position_size * COMMISSION * 2  # Покупка + продажа
            net_pnl = gross_pnl - commission_cost
            
            period_pnl += net_pnl
            
            period_trades.append({
                'Date': entry_date,
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
        
        positions_history.append({
            'Date': entry_date,
            'Symbols': top_symbols,
            'Probabilities': [top_probs[s] for s in top_symbols]
        })
        
        print(f"📅 {date_str}: Capital: ${capital:,.2f} | PnL: ${period_pnl:+.2f} | Позиций: {len(period_trades)}")
    
    # Создаем DataFrame
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trades_df = pd.DataFrame(trades)
    
    # Рассчитываем метрики
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    total_return_pct = total_return * 100
    
    # Годовая доходность (аннуализируем)
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days
    n_years = n_days / 365
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Ежедневные доходности для Sharpe
    daily_returns = equity_df['Capital'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Максимальная просадка
    rolling_max = equity_df['Capital'].cummax()
    drawdown = (equity_df['Capital'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = trades_df[trades_df['Net PnL ($)'] > 0]
    losing_trades = trades_df[trades_df['Net PnL ($)'] <= 0]
    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
    
    # Profit Factor
    total_gross_profit = winning_trades['Net PnL ($)'].sum() if len(winning_trades) > 0 else 0
    total_gross_loss = abs(losing_trades['Net PnL ($)'].sum()) if len(losing_trades) > 0 else 1
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0
    
    # Buy-and-Hold за период
    bh_returns = {}
    for symbol in price_pivot.columns:
        if START_DATE in price_pivot.index.strftime('%Y-%m-%d').tolist() and END_DATE in price_pivot.index.strftime('%Y-%m-%d').tolist():
            start_price = price_pivot.loc[START_DATE, symbol] if symbol in price_pivot.columns else np.nan
            end_price = price_pivot.loc[END_DATE, symbol] if symbol in price_pivot.columns else np.nan
            if not np.isnan(start_price) and not np.isnan(end_price):
                bh_returns[symbol] = (end_price / start_price) - 1
    
    bh_return = np.mean(list(bh_returns.values())) if bh_returns else 0
    bh_capital = INITIAL_CAPITAL * (1 + bh_return)
    alpha = total_return - bh_return
    
    # Вывод результатов
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*70)
    print(f"Период:               {START_DATE} → {END_DATE}")
    print(f"Стартовый капитал:    ${INITIAL_CAPITAL:>10,.0f}")
    print(f"Финальный капитал:    ${capital:>10,.2f}")
    print("─"*70)
    print(f"Суммарная доходность: {total_return:>+10.2%}")
    print(f"Годовая доходность:   {annual_return:>+10.2%}")
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
    
    # Оценка стратегии
    print(f"\n=== ОЦЕНКА СТРАТЕГИИ ({START_DATE} → {END_DATE}) ===\n")
    
    metrics = {
        'Sharpe Ratio':   (sharpe_ratio,     1.0,   0.5,  '{:.2f}'),
        'Макс. просадка': (max_drawdown,    -0.15, -0.30, '{:.2%}'),
        'Profit Factor':  (profit_factor,    1.3,   1.0,  '{:.2f}'),
        'Win Rate':       (win_rate,         0.55,  0.45, '{:.1%}'),
        'Alpha vs B&H':   (alpha,            0.0,  -0.10, '{:.2%}'),
    }
    
    print(f"{'Метрика':<20} {'Значение':>10}  {'Цель':>8}  {'Стоп':>8}  Статус")
    print('─' * 65)
    for name, (val, good, bad, fmt) in metrics.items():
        if name == 'Макс. просадка':
            status = '✅ Хорошо' if val > good else ('❌ Плохо' if val < bad else '⚠️  Терпимо')
        else:
            status = '✅ Хорошо' if val >= good else ('❌ Плохо' if val < bad else '⚠️  Терпимо')
        print(f"{name:<20} {fmt.format(val):>10}  {fmt.format(good):>8}  {fmt.format(bad):>8}  {status}")
    print('─' * 65)
    
    # Итоговый вердикт
    n_good = sum(1 for name, (val, good, bad, _) in metrics.items()
                 if (val > good if name == 'Макс. просадка' else val >= good))
    n_bad = sum(1 for name, (val, good, bad, _) in metrics.items()
                if (val < bad if name == 'Макс. просадка' else val < bad))
    
    print(f"\nИтоговый вердикт:")
    if n_bad == 0 and n_good >= 3:
        print("  ✅ Стратегия готова к paper trading")
    elif n_bad <= 1:
        print("  ⚠️  Стратегия требует доработки перед реальной торговлей")
    else:
        print("  ❌ Стратегия не готова к реальной торговле")
    
    # Визуализация
    create_visualizations(equity_df, trades_df, drawdown, bh_capital, total_return, max_drawdown, profit_factor)
    
    # Топ сделок
    if len(trades_df) > 0:
        print("\n=== Топ-10 ЛУЧШИХ СДЕЛОК ===")
        print(trades_df.nlargest(10, 'Net PnL ($)')[['Date', 'Symbol', 'Probability', 'Forward Return (10d)', 'Net PnL ($)']].to_string(index=False))
        
        print("\n=== Топ-10 ХУДШИХ СДЕЛОК ===")
        print(trades_df.nsmallest(10, 'Net PnL ($)')[['Date', 'Symbol', 'Probability', 'Forward Return (10d)', 'Net PnL ($)']].to_string(index=False))
    
    # Сохраняем результаты
    save_results(equity_df, trades_df, positions_history)
    
    return equity_df, trades_df


def create_visualizations(equity_df, trades_df, drawdown, bh_capital, total_return, max_drawdown, profit_factor):
    """Создаёт графики"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Кривая капитала
    ax1 = axes[0, 0]
    ax1.plot(equity_df.index, equity_df['Capital'], color='#9b59b6', linewidth=2, label='Стратегия')
    ax1.axhline(INITIAL_CAPITAL, color='grey', linestyle='--', alpha=0.5, label='Стартовый капитал')
    ax1.axhline(bh_capital, color='#3498db', linestyle='--', alpha=0.7, label=f'Buy-and-Hold')
    ax1.fill_between(equity_df.index, INITIAL_CAPITAL, equity_df['Capital'],
                     where=equity_df['Capital'] >= INITIAL_CAPITAL, alpha=0.15, color='green')
    ax1.fill_between(equity_df.index, INITIAL_CAPITAL, equity_df['Capital'],
                     where=equity_df['Capital'] < INITIAL_CAPITAL, alpha=0.15, color='red')
    ax1.set_title(f'Кривая капитала | Итог: ${equity_df["Capital"].iloc[-1]:,.0f} ({total_return:+.1%})', fontweight='bold')
    ax1.set_ylabel('Капитал ($)')
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(alpha=0.3)
    
    # 2. Просадка
    ax2 = axes[0, 1]
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='#9b59b6', alpha=0.4)
    ax2.axhline(-0.15, color='orange', linestyle='--', alpha=0.7, label='Цель -15%')
    ax2.axhline(-0.30, color='red', linestyle='--', alpha=0.7, label='Стоп -30%')
    ax2.set_title(f'Просадка | Макс: {max_drawdown:.2%}', fontweight='bold')
    ax2.set_ylabel('Просадка (%)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. P&L по периодам
    ax3 = axes[1, 0]
    period_pnl = equity_df['PnL']
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in period_pnl]
    ax3.bar(period_pnl.index, period_pnl.values, color=colors, alpha=0.7, width=REBALANCE_DAYS-1)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('P&L по торговым периодам', fontweight='bold')
    ax3.set_ylabel('P&L ($)')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Распределение P&L сделок
    ax4 = axes[1, 1]
    if len(trades_df) > 0:
        ax4.hist(trades_df['Net PnL ($)'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(trades_df['Net PnL ($)'].mean(), color='green', linestyle='--', alpha=0.7, label=f'Среднее: ${trades_df["Net PnL ($)"].mean():.2f}')
        ax4.set_title(f'Распределение P&L сделок | Profit Factor: {profit_factor:.2f}', fontweight='bold')
        ax4.set_xlabel('P&L ($)')
        ax4.set_ylabel('Частота')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # 5. Топ акций по частоте
    ax5 = axes[2, 0]
    if len(trades_df) > 0:
        top_symbols = trades_df['Symbol'].value_counts().head(10)
        ax5.barh(range(len(top_symbols)), top_symbols.values, color='#3498db', alpha=0.7)
        ax5.set_yticks(range(len(top_symbols)))
        ax5.set_yticklabels(top_symbols.index)
        ax5.set_title('Топ-10 акций по частоте сделок', fontweight='bold')
        ax5.set_xlabel('Количество сделок')
        ax5.grid(alpha=0.3, axis='x')
    
    # 6. Средняя доходность по акциям
    ax6 = axes[2, 1]
    if len(trades_df) > 0:
        symbol_returns = trades_df.groupby('Symbol')['Forward Return (10d)'].mean().sort_values(ascending=False).head(10)
        colors_ret = ['#2ecc71' if x > 0 else '#e74c3c' for x in symbol_returns.values]
        ax6.barh(range(len(symbol_returns)), symbol_returns.values, color=colors_ret, alpha=0.7)
        ax6.set_yticks(range(len(symbol_returns)))
        ax6.set_yticklabels(symbol_returns.index)
        ax6.set_title('Топ-10 акций по средней доходности', fontweight='bold')
        ax6.set_xlabel('Средняя доходность (%)')
        ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax6.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Сохраняем график
    fig.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Графики сохранены в backtest_results.png")


def save_results(equity_df, trades_df, positions_history):
    """Сохраняет результаты"""
    output_path = Path(__file__).parent / "backtest_full_report.json"
    
    report = {
        'equity_curve': equity_df.to_dict(orient='records'),
        'trades': trades_df.to_dict(orient='records') if len(trades_df) > 0 else [],
        'positions_history': positions_history,
        'summary': {
            'final_capital': float(equity_df.iloc[-1]['Capital']),
            'total_return_pct': float((equity_df.iloc[-1]['Capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100),
            'total_trades': len(trades_df),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Сохраняем CSV
    equity_df.to_csv('backtest_equity_full.csv')
    if len(trades_df) > 0:
        trades_df.to_csv('backtest_trades_full.csv', index=False)
    
    print(f"\n💾 Результаты сохранены:")
    print(f"   - backtest_full_report.json")
    print(f"   - backtest_equity_full.csv")
    if len(trades_df) > 0:
        print(f"   - backtest_trades_full.csv")


if __name__ == "__main__":
    print("\n" + "🎯"*35)
    print("БЭКТЕСТ СТРАТЕГИИ ПРОГНОЗИРОВАНИЯ S&P 500")
    print("🎯"*35)
    
    # Сначала генерируем прогнозы
    print("\nШаг 1: Генерация исторических прогнозов...")
    exec(open("generate_daily_predictions.py").read())
    
    # Запускаем бэктест
    print("\nШаг 2: Запуск бэктеста...")
    equity_df, trades_df = run_backtest()
    
    print("\n✅ Бэктест завершён!")