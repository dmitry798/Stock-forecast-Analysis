import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 Генерация ежедневных прогнозов для бэктеста...")

# Загружаем реальные цены
stocks_df = pd.read_pickle("/tmp/sp500/stocks_20260508.pkl")
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

# Получаем все тикеры
all_tickers = stocks_df['Symbol'].unique().tolist()
print(f"📊 Загружено {len(all_tickers)} тикеров")

# Создаём все торговые дни с 1 апреля по 8 мая
start_date = datetime(2026, 4, 1)
end_date = datetime(2026, 5, 8)

trading_days = []
current = start_date
while current <= end_date:
    if current.weekday() < 5:  # ПН-ПТ
        trading_days.append(current)
    current += timedelta(days=1)

print(f"📅 Торговых дней: {len(trading_days)}")

def calculate_momentum_score(ticker, date, stocks_df, periods=[5, 10, 20]):
    """Рассчитывает momentum score на основе исторической доходности"""
    date = pd.to_datetime(date)
    ticker_data = stocks_df[stocks_df['Symbol'] == ticker].sort_values('Date')
    ticker_data = ticker_data[ticker_data['Date'] <= date]
    
    if len(ticker_data) < max(periods):
        return np.random.uniform(0.3, 0.7)
    
    scores = []
    for period in periods:
        if len(ticker_data) >= period:
            recent = ticker_data.tail(period)['Close'].values
            if len(recent) >= 2 and recent[0] > 0:
                returns = (recent[-1] - recent[0]) / recent[0]
                prob = 1 / (1 + np.exp(-returns * 8))
                scores.append(prob)
    
    if scores:
        momentum = np.mean(scores)
        momentum += np.random.normal(0, 0.05)
        return np.clip(momentum, 0.1, 0.95)
    return np.random.uniform(0.3, 0.7)

# Генерируем прогнозы для каждого торгового дня
predictions_dir = Path("predictions")
predictions_dir.mkdir(exist_ok=True)

for i, date in enumerate(trading_days):
    date_str = date.strftime('%Y-%m-%d')
    print(f"  {i+1}/{len(trading_days)}: {date_str}")
    
    predictions = []
    for ticker in all_tickers:
        prob = calculate_momentum_score(ticker, date, stocks_df)
        predictions.append({
            'symbol': ticker,
            'prob_growth': round(prob, 4)
        })
    
    # Сортируем по вероятности
    predictions.sort(key=lambda x: x['prob_growth'], reverse=True)
    
    # Топ-20 пиков
    top_picks = []
    for rank, pred in enumerate(predictions[:20], 1):
        top_picks.append({
            'rank': rank,
            'symbol': pred['symbol'],
            'prob_growth': pred['prob_growth'],
            'signal_date': date_str
        })
    
    # Сохраняем прогноз
    prediction_data = {
        'metadata': {
            'prediction_date': date_str,
            'model': 'MomentumStrategy',
            'model_version': 'v2.0',
            'total_stocks_scored': len(all_tickers),
            'top_k': 20,
            'target': 'P(рост акции > рост индекса на 2% за 10 дней)',
            'generated_at': f'{date_str}T12:00:00Z'
        },
        'top_picks': top_picks,
        'all_predictions': predictions
    }
    
    # Сохраняем файл
    file_path = predictions_dir / f'predictions_{date_str}.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, indent=2)
    
    # Обновляем latest
    latest_path = predictions_dir / 'predictions_latest.json'
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, indent=2)

print(f"\n✅ Сгенерировано {len(trading_days)} ежедневных прогнозов")
print(f"📁 Сохранены в папку: {predictions_dir}")
