# 🚀 Smart Money Futures Signal Bot

Advanced cryptocurrency futures trading signal bot with ML-based analysis, smart signal cancellation, and automated trading on BingX.

## 📊 Overview

This bot analyzes **11 trading pairs** (BTC, ETH, BNB, SOL, AVAX, DOGE, LINK, XRP, TRX, ADA, HYPE) using multiple market indicators on 5-minute candles to generate high-confidence trading signals delivered via Telegram.

### Key Features

- 🧠 **Enhanced Formula v2** - ML-based profit prediction (Random Forest Regressor)
- 🎯 **Smart Signal Cancellation** - Automatically cancels signals when market conditions deteriorate
- 📈 **Dual-Strategy Scoring** - Weighted algorithm with custom indicator weights per coin
- 💹 **Automated Trading** - BingX integration with robust risk management
- 🔍 **Order Flow Analysis** - Psychological levels and bid-ask aggression detection
- 🤖 **AI Analyst Service** - OpenAI-powered independent signal analysis
- ⚡ **Real-time CVD & Liquidations** - Binance WebSocket data streams

## 🎯 Performance

Based on comprehensive backtesting over 7 days (2,762 signals):

- **BUY Signals**: 62.3% win rate (with positive CVD)
- **SELL Signals**: 47.6% win rate (with negative CVD)
- **Expected Returns**: +$2,170/week (+$112,865/year)
- **Risk Management**: 24% TP, 4% SL, 71% TTL exits (6:1 TP/SL ratio)

### Optimal Trading Parameters

- **Stop-Loss**: 10% of position size (0.20% price movement at 50x leverage)
- **Take-Profit Strategy**: Hybrid (BUY=conservative, SELL=aggressive)
- **Position Exit**: TP, SL, or TTL only

## 🏗️ Architecture

### Core Services

1. **Smart Money Signal Bot** (`main.py`) - Main signal generation engine
2. **Signal Tracker** (`signal_tracker.py`) - Tracks signal effectiveness and cancellations
3. **CVD Service** (`cvd_service.py`) - Cumulative Volume Delta from Binance WebSocket
4. **Liquidation Service** (`liquidation_service.py`) - Tracks liquidation events
5. **BingX Auto-Trader** (`bingx_trader_service.py`) - Automated position management
6. **AI Analyst** (`services/ai_analyst/`) - OpenAI-powered market analysis
7. **Data Feeds** (`services/data_feeds/`) - Comprehensive market data collection
8. **UIF Feature Engine** (`services/uif_feature_engine/`) - Technical indicators for ML

### Smart Signal Cancellation

Signals are automatically cancelled when:

1. **Confidence drops below 30%** (absolute threshold)
2. **Opposite signal appears** with ≥70% confidence
3. **Regime shift against signal**:
   - BUY cancelled: Bull→Bear transition OR Neutral→Bearish
   - SELL cancelled: Bear→Bull transition OR Neutral→Bullish

## 📡 Data Sources

- **Coinalyze API** - Aggregated futures data (OHLCV, Open Interest)
- **Binance WebSocket** - Real-time CVD and Liquidations
- **CryptoCompare API** - UIF Feature Engine OHLCV
- **OKX API** - Funding rate data
- **OpenAI API** - AI Analyst market context (gpt-4o-mini)

## 🔧 Setup

### Prerequisites

- Python 3.10+
- Replit environment (or Linux with NixOS)

### Required Secrets

Set the following environment secrets in Replit:

```bash
COINALYZE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TRADING_TELEGRAM_BOT_TOKEN=your_trading_bot_token
```

### Installation

1. Clone the repository
2. Install dependencies (handled automatically by Replit)
3. Configure environment secrets
4. Start workflows via Replit interface

### Workflows

The bot runs 9 parallel services:

- Smart Money Signal Bot
- Signal Tracker
- CVD Service
- Liquidation Service
- BingX Auto-Trader
- AI Analyst + Commands
- Data Feeds Service
- UIF Feature Engine
- Watchdog Monitor

## 📊 Signal Format

Telegram signals include:

- Entry price and direction (BUY/SELL)
- Target zone (min-max range)
- Confidence percentage
- Time-To-Live (TTL) duration
- Market strength indicators (CVD, OI, Funding)
- Regime classification (Bull/Bear/Neutral)

## 🛡️ Risk Management

- **Position Sizing**: Configurable per signal
- **Stop-Loss**: 10% of position (validated by backtesting)
- **Take-Profit**: Hybrid strategy optimized for signal type
- **Smart Cancellation**: Protects against regime reversals
- **Fee Optimization**: Limit orders for TP (0.02% maker fee)

## 📈 Advanced Features

### Hybrid EMA+VWAP Regime Detection

6-regime system combining fast EMA reaction with institutional VWAP reference:
- Strong Bull / Bull / Neutral / Bear / Strong Bear / Sideways

### Order Flow Indicators

- **Psychological Level Detector**: +20.6% win rate improvement near round numbers
- **Bid-Ask Aggression**: CVD-based buyer/seller strength analysis

### Self-Learning Controller

- Wilson score method for weight optimization
- Logistic regression for ML-optimized indicator weights
- Continuous learning from historical signals

## 📁 Project Structure

```
.
├── main.py                      # Main signal generation
├── signals/                     # Signal logic (features, scoring, formatting)
├── signal_tracker.py            # Effectiveness tracking
├── cvd_service.py              # CVD data collection
├── liquidation_service.py      # Liquidation tracking
├── bingx_trader_service.py     # Auto-trading service
├── watchdog.py                 # System monitoring
├── services/
│   ├── ai_analyst/            # AI-powered analysis
│   ├── data_feeds/            # Market data collection
│   └── uif_feature_engine/    # Technical indicators
├── backtesting/               # Backtest scripts
├── analysis/                  # Performance reports
└── configs/                   # Configuration files
```

## 🤝 Contributing

This is a production trading system. Major changes should be discussed before implementation.

## ⚠️ Disclaimer

This bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 📝 License

Proprietary - All rights reserved

---

**Built with ❤️ on Replit**
