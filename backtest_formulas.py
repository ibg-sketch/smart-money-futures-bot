#!/usr/bin/env python3
"""
BACKTEST ФОРМУЛ НА СЫРЫХ РЫНОЧНЫХ ДАННЫХ

Анализирует эффективность формул расчета целей (target_min/max) 
на реальных ценовых движениях за последние несколько дней.

НЕ ИСПОЛЬЗУЕТ готовые сигналы - применяет формулы к каждой свече!
"""

import requests
import pandas as pd
import numpy as np
import yaml
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Импортируем функции расчета из основного бота
from signals.features import calculate_atr
from signals.scoring import calculate_price_targets

# Coinalyze API
COINALYZE_API_KEY = os.getenv('COINALYZE_API_KEY')
COINALYZE_API = "https://api.coinalyze.net/v1"

# Символы для тестирования
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AVAXUSDT',
    'DOGEUSDT', 'LINKUSDT', 'XRPUSDT', 'TRXUSDT', 'ADAUSDT', 'HYPEUSDT'
]

INTERVAL = '15m'  # 15-минутные свечи
LOOKBACK_DAYS = 3  # Последние 3 дня

def symbol_to_coinalyze(symbol):
    """Convert BTCUSDT -> BTCUSD.X or BTCUSDPERP.X"""
    base = symbol.replace('USDT', '')
    # Попробуем оба варианта
    return f"{base}USD.6,{base}USDPERP.6"

def fetch_coinalyze_klines(symbol, interval='15min', limit=300):
    """Загрузить исторические свечи с Coinalyze API"""
    to_ts = int(time.time())
    
    # Рассчитать from_ts на основе лимита
    minutes_map = {
        '1min': 1, '3min': 3, '5min': 5, '15min': 15, 
        '30min': 30, '1hour': 60, '4hour': 240
    }
    minutes = minutes_map.get(interval, 15)
    from_ts = to_ts - (limit * minutes * 60)
    
    symbols_param = symbol_to_coinalyze(symbol)
    
    url = f"{COINALYZE_API}/ohlcv-history"
    params = {
        'symbols': symbols_param,
        'interval': interval,
        'from': from_ts,
        'to': to_ts
    }
    
    headers = {
        'api-key': COINALYZE_API_KEY
    } if COINALYZE_API_KEY else {}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                hist = data[0].get('history', [])
                if hist:
                    # Конвертируем в формат Binance [timestamp, o, h, l, c, v]
                    klines = [
                        [
                            int(h['t']) * 1000,  # timestamp в мс
                            float(h['o']),       # open
                            float(h['h']),       # high
                            float(h['l']),       # low
                            float(h['c']),       # close
                            float(h.get('v', 0)) # volume
                        ]
                        for h in hist
                    ]
                    return klines
        else:
            print(f"⚠️ Coinalyze API error {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
    
    return None

def calculate_future_max_min(klines, start_idx, lookforward_candles=2):
    """
    Рассчитать максимум и минимум цены в следующих N свечах
    
    Args:
        klines: Список свечей [timestamp, o, h, l, c, v, ...]
        start_idx: Индекс текущей свечи
        lookforward_candles: Сколько свечей вперед смотреть (2 = 30 мин для 15m)
    
    Returns:
        (future_high_pct, future_low_pct) - процентное изменение от текущей цены
    """
    if start_idx >= len(klines) - 1:
        return None, None
    
    current_close = float(klines[start_idx][4])
    
    # Собираем high/low из следующих свечей
    end_idx = min(start_idx + 1 + lookforward_candles, len(klines))
    future_candles = klines[start_idx + 1:end_idx]
    
    if not future_candles:
        return None, None
    
    future_highs = [float(c[2]) for c in future_candles]
    future_lows = [float(c[3]) for c in future_candles]
    
    max_high = max(future_highs)
    min_low = min(future_lows)
    
    high_pct = ((max_high - current_close) / current_close) * 100
    low_pct = ((min_low - current_close) / current_close) * 100
    
    return high_pct, low_pct

def apply_formula_to_candle(klines, idx, symbol, config):
    """
    Применить формулу расчета целей к одной свече
    
    Returns:
        dict с расчетами или None если недостаточно данных
    """
    if idx < 15:  # Нужно минимум 15 свечей для ATR
        return None
    
    # Берем свечи до текущей (для расчета индикаторов)
    hist_klines = klines[max(0, idx-200):idx+1]
    
    current_candle = klines[idx]
    timestamp = current_candle[0]
    price = float(current_candle[4])  # Close price
    
    # Рассчитываем ATR
    atr = calculate_atr(hist_klines, period=14)
    if atr is None or atr <= 0:
        return None
    
    # Имитируем минимальные данные для формулы
    # (в реальности нужны CVD, OI, Volume - здесь используем нейтральные значения)
    volume_data = {
        'last': float(current_candle[5]),
        'median': float(current_candle[5]),  # Нейтральное значение
        'oi_current': 1_000_000_000
    }
    
    coin_config = config.get(symbol, {})
    
    # Применяем формулу ДЛЯ BUY (тестируем восходящее движение)
    try:
        results_buy = calculate_price_targets(
            price=price,
            confidence=0.5,  # Нейтральное
            cvd=0,  # Нейтральное (без CVD влияния)
            symbol=symbol,
            coin_config=coin_config,
            klines=hist_klines,
            volume_data=volume_data,
            oi_change=0,  # Нейтральное
            verdict='BUY',
            vwap=price  # Нейтральное (цена = VWAP)
        )
        
        target_min_buy, target_max_buy = results_buy[0], results_buy[1]
        multiplier_buy = results_buy[4]
        
    except Exception as e:
        print(f"⚠️ Formula error for {symbol} BUY: {e}")
        return None
    
    # Применяем формулу ДЛЯ SELL (тестируем нисходящее движение)
    try:
        results_sell = calculate_price_targets(
            price=price,
            confidence=0.5,
            cvd=0,
            symbol=symbol,
            coin_config=coin_config,
            klines=hist_klines,
            volume_data=volume_data,
            oi_change=0,
            verdict='SELL',
            vwap=price
        )
        
        target_min_sell, target_max_sell = results_sell[0], results_sell[1]
        multiplier_sell = results_sell[4]
        
    except Exception as e:
        print(f"⚠️ Formula error for {symbol} SELL: {e}")
        return None
    
    # Рассчитываем реальное движение цены в следующие 30 мин (2 свечи по 15m)
    future_high_pct, future_low_pct = calculate_future_max_min(klines, idx, lookforward_candles=2)
    
    if future_high_pct is None or future_low_pct is None:
        return None
    
    return {
        'timestamp': datetime.fromtimestamp(timestamp / 1000),
        'symbol': symbol,
        'price': price,
        'atr': atr,
        'atr_pct': (atr / price) * 100,
        
        # BUY формула
        'target_min_buy': target_min_buy,
        'target_max_buy': target_max_buy,
        'multiplier_buy': multiplier_buy,
        
        # SELL формула
        'target_min_sell': target_min_sell,
        'target_max_sell': target_max_sell,
        'multiplier_sell': multiplier_sell,
        
        # Реальное движение
        'future_high_pct': future_high_pct,
        'future_low_pct': future_low_pct,
        
        # Проверка достижимости
        'buy_min_hit': future_high_pct >= target_min_buy,
        'buy_max_hit': future_high_pct >= target_max_buy,
        'sell_min_hit': abs(future_low_pct) >= target_min_sell,
        'sell_max_hit': abs(future_low_pct) >= target_max_sell,
    }

def analyze_symbol(symbol, config):
    """Анализ одной монеты"""
    print(f"\n{'='*60}")
    print(f"📊 Analyzing {symbol}...")
    print(f"{'='*60}")
    
    # Загрузить данные (Coinalyze API, последние 300 свечей = ~75 часов для 15min)
    klines = fetch_coinalyze_klines(symbol, interval='15min', limit=300)
    
    if not klines or len(klines) < 20:
        print(f"❌ Insufficient data for {symbol}")
        return None
    
    print(f"✅ Loaded {len(klines)} candles")
    print(f"   Date range: {datetime.fromtimestamp(klines[0][0]/1000)} → {datetime.fromtimestamp(klines[-1][0]/1000)}")
    
    # Применяем формулу к каждой свече
    results = []
    for idx in range(15, len(klines) - 2):  # -2 чтобы было место для future lookforward
        result = apply_formula_to_candle(klines, idx, symbol, config)
        if result:
            results.append(result)
    
    if not results:
        print(f"❌ No valid results for {symbol}")
        return None
    
    df = pd.DataFrame(results)
    print(f"✅ Analyzed {len(df)} candles")
    
    # Статистика
    buy_min_rate = df['buy_min_hit'].mean() * 100
    buy_max_rate = df['buy_max_hit'].mean() * 100
    sell_min_rate = df['sell_min_hit'].mean() * 100
    sell_max_rate = df['sell_max_hit'].mean() * 100
    
    avg_target_buy_min = df['target_min_buy'].mean()
    avg_target_buy_max = df['target_max_buy'].mean()
    avg_target_sell_min = df['target_min_sell'].mean()
    avg_target_sell_max = df['target_max_sell'].mean()
    
    avg_future_high = df['future_high_pct'].mean()
    avg_future_low = df['future_low_pct'].abs().mean()
    
    print(f"\n📈 BUY TARGETS (upward movement):")
    print(f"   Target MIN: {avg_target_buy_min:.2f}% | Hit rate: {buy_min_rate:.1f}%")
    print(f"   Target MAX: {avg_target_buy_max:.2f}% | Hit rate: {buy_max_rate:.1f}%")
    print(f"   Avg real upward move: {avg_future_high:.2f}%")
    
    print(f"\n📉 SELL TARGETS (downward movement):")
    print(f"   Target MIN: {avg_target_sell_min:.2f}% | Hit rate: {sell_min_rate:.1f}%")
    print(f"   Target MAX: {avg_target_sell_max:.2f}% | Hit rate: {sell_max_rate:.1f}%")
    print(f"   Avg real downward move: {avg_future_low:.2f}%")
    
    return df

def main():
    """Главная функция анализа"""
    print("="*80)
    print("🔬 BACKTEST ФОРМУЛ НА СЫРЫХ РЫНОЧНЫХ ДАННЫХ")
    print("="*80)
    print(f"Период: последние {LOOKBACK_DAYS} дня")
    print(f"Интервал: {INTERVAL} (15 минут)")
    print(f"Символы: {len(SYMBOLS)} монет")
    print(f"Окно проверки: 30 минут вперед (2 свечи)")
    print("="*80)
    
    # Загрузить конфигурацию
    with open('config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)
    
    config = full_config.get('coins', {})
    
    # Анализировать каждую монету
    all_results = []
    
    for symbol in SYMBOLS:
        df = analyze_symbol(symbol, config)
        if df is not None:
            all_results.append(df)
        time.sleep(0.5)  # Rate limiting
    
    if not all_results:
        print("\n❌ No results to analyze")
        return
    
    # Объединенная статистика
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print("\n" + "="*80)
    print("📊 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ МОНЕТАМ")
    print("="*80)
    
    total_candles = len(combined_df)
    
    # BUY статистика
    buy_min_rate = combined_df['buy_min_hit'].mean() * 100
    buy_max_rate = combined_df['buy_max_hit'].mean() * 100
    avg_buy_min = combined_df['target_min_buy'].mean()
    avg_buy_max = combined_df['target_max_buy'].mean()
    avg_multiplier_buy = combined_df['multiplier_buy'].mean()
    
    # SELL статистика
    sell_min_rate = combined_df['sell_min_hit'].mean() * 100
    sell_max_rate = combined_df['sell_max_hit'].mean() * 100
    avg_sell_min = combined_df['target_min_sell'].mean()
    avg_sell_max = combined_df['target_max_sell'].mean()
    avg_multiplier_sell = combined_df['multiplier_sell'].mean()
    
    # Реальные движения
    avg_real_up = combined_df['future_high_pct'].mean()
    avg_real_down = combined_df['future_low_pct'].abs().mean()
    
    print(f"\n📊 Всего проанализировано свечей: {total_candles:,}")
    print(f"   Средняя волатильность (ATR%): {combined_df['atr_pct'].mean():.3f}%")
    
    print(f"\n🟢 BUY ФОРМУЛА (восходящее движение):")
    print(f"   Средний target_min: {avg_buy_min:.2f}%  →  Hit rate: {buy_min_rate:.1f}%")
    print(f"   Средний target_max: {avg_buy_max:.2f}%  →  Hit rate: {buy_max_rate:.1f}%")
    print(f"   Средний multiplier: {avg_multiplier_buy:.2f}")
    print(f"   Реальное движение вверх: {avg_real_up:.2f}%")
    
    print(f"\n🔴 SELL ФОРМУЛА (нисходящее движение):")
    print(f"   Средний target_min: {avg_sell_min:.2f}%  →  Hit rate: {sell_min_rate:.1f}%")
    print(f"   Средний target_max: {avg_sell_max:.2f}%  →  Hit rate: {sell_max_rate:.1f}%")
    print(f"   Средний multiplier: {avg_multiplier_sell:.2f}")
    print(f"   Реальное движение вниз: {avg_real_down:.2f}%")
    
    # Оценка калибровки
    print(f"\n🎯 КАЛИБРОВКА ФОРМУЛ:")
    buy_min_calibration = avg_real_up / avg_buy_min if avg_buy_min > 0 else 0
    sell_min_calibration = avg_real_down / avg_sell_min if avg_sell_min > 0 else 0
    
    print(f"   BUY: реальное движение / target_min = {buy_min_calibration:.2f}x")
    print(f"   SELL: реальное движение / target_min = {sell_min_calibration:.2f}x")
    
    if buy_min_calibration > 1.5:
        print(f"   ⚠️ BUY targets слишком консервативные (цена движется сильнее на {(buy_min_calibration-1)*100:.0f}%)")
    elif buy_min_calibration < 0.8:
        print(f"   ⚠️ BUY targets слишком агрессивные (цена не доходит на {(1-buy_min_calibration)*100:.0f}%)")
    else:
        print(f"   ✅ BUY targets хорошо откалиброваны")
    
    if sell_min_calibration > 1.5:
        print(f"   ⚠️ SELL targets слишком консервативные (цена движется сильнее на {(sell_min_calibration-1)*100:.0f}%)")
    elif sell_min_calibration < 0.8:
        print(f"   ⚠️ SELL targets слишком агрессивные (цена не доходит на {(1-sell_min_calibration)*100:.0f}%)")
    else:
        print(f"   ✅ SELL targets хорошо откалиброваны")
    
    # Сохранить результаты
    output_file = 'formula_backtest_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n💾 Результаты сохранены: {output_file}")
    
    # Дополнительная статистика по квартилям multiplier
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ MULTIPLIER:")
    
    print(f"\n   BUY multiplier квартили:")
    q_buy = combined_df['multiplier_buy'].quantile([0.25, 0.5, 0.75])
    print(f"      Q1 (25%): {q_buy[0.25]:.2f}")
    print(f"      Q2 (50%): {q_buy[0.50]:.2f}")
    print(f"      Q3 (75%): {q_buy[0.75]:.2f}")
    
    print(f"\n   SELL multiplier квартили:")
    q_sell = combined_df['multiplier_sell'].quantile([0.25, 0.5, 0.75])
    print(f"      Q1 (25%): {q_sell[0.25]:.2f}")
    print(f"      Q2 (50%): {q_sell[0.50]:.2f}")
    print(f"      Q3 (75%): {q_sell[0.75]:.2f}")
    
    print("\n" + "="*80)
    print("✅ Анализ завершён!")
    print("="*80)

if __name__ == "__main__":
    main()
