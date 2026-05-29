# Sentiment Alpha Trader

A systematic **long/short equity trading system** that uses NLP to find alpha in financial news. Every morning before market open, it scores news headlines for 25 S&P 100 stocks using **FinBERT** (a finance-trained BERT model), generates buy/sell signals, and executes trades automatically via the **Alpaca brokerage API**.

---

## What It Does

1. **Fetches** recent news headlines for each ticker from NewsAPI
2. **Scores** each headline with FinBERT — a transformer model fine-tuned on financial text — producing a sentiment score from -1 (negative) to +1 (positive)
3. **Generates signals** — tickers above +0.35 go LONG, below -0.35 go SHORT, rest stay FLAT
4. **Sizes positions** equally across up to 10 active trades (5% cash buffer)
5. **Executes** market-on-open orders via Alpaca's paper trading API at 9:28 AM ET daily
6. **Monitors** everything through a live dashboard

---

## Tech Stack

| Layer         | Tool                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| NLP Model     | [FinBERT (ProsusAI)](https://huggingface.co/ProsusAI/finbert) via HuggingFace |
| News Data     | [NewsAPI](https://newsapi.org)                                                |
| Brokerage API | [Alpaca Markets](https://alpaca.markets) (paper trading)                      |
| Price Data    | [yfinance](https://github.com/ranaroussi/yfinance)                            |
| Scheduler     | Python `schedule` + cron                                                      |
| Dashboard     | Vanilla HTML/CSS/JS                                                           |
| Language      | Python 3.11+                                                                  |

---

## Project Structure

```
sentiment-trader/
├── core/
│   ├── sentiment.py     # NewsAPI fetching + FinBERT scoring
│   ├── signals.py       # Signal generation + position sizing
│   ├── execution.py     # Alpaca order execution + account management
│   └── backtest.py      # Vectorized historical backtest engine
├── dashboard/
│   └── index.html       # Live monitoring dashboard (open in browser)
├── data/                # Signal CSVs + equity curve saved here
├── results/
│   └── backtest_results.png
├── main.py              # Entry point (dry-run / live / backtest)
├── scheduler.py         # Fires pipeline at 9:28 AM ET on weekdays
├── requirements.txt
└── .env.example         # API key template
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/kashishbalan/sentiment-trader.git
cd sentiment-trader
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get API keys (both free)

- **NewsAPI** → [newsapi.org](https://newsapi.org) — click Get API Key (100 req/day free)
- **Alpaca** → [alpaca.markets](https://alpaca.markets) — sign up, go to Paper Trading, generate API keys

### 3. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

```
NEWSAPI_KEY=your_key
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Run

```bash
# Test it — scores tickers and prints signals, no trades placed
python main.py --mode dry-run

# Paper trade — executes real orders on Alpaca paper account
python main.py --mode live

# Open the dashboard
open dashboard/index.html
```

### 5. Schedule (optional)

```bash
# Runs automatically at 9:28 AM ET every weekday
python scheduler.py
```

---

## Backtest Results

Simulated on S&P 100 subset, Jan 2022 – Dec 2023, equal-weight sizing, no transaction costs.

| Metric                | Value      |
| --------------------- | ---------- |
| Annualized Return     | +20.2%     |
| Annualized Volatility | 17.5%      |
| Sharpe Ratio          | 1.15       |
| Max Drawdown          | -8.3%      |
| Win Rate              | 54.2%      |
| Universe              | 25 tickers |

---

## How the Signal Works

```
NewsAPI headlines (last 24h, per ticker)
          ↓
   FinBERT scoring per headline  →  score ∈ [-1, +1]
          ↓
   Mean score across all headlines
          ↓
   score ≥ +0.35  →  LONG
   score ≤ -0.35  →  SHORT
   otherwise      →  FLAT
          ↓
   Top 10 by |score|, equal dollar sizing
          ↓
   Market-on-open orders via Alpaca
```

FinBERT is used over generic sentiment tools (VADER, TextBlob) because it's trained specifically on financial text — it understands that _"beats estimates"_ is positive and _"misses guidance"_ is negative in context.

---

## Tuning Parameters

In `core/signals.py`:

```python
LONG_THRESHOLD  = 0.35   # raise to trade only on strong sentiment
SHORT_THRESHOLD = 0.35
MAX_POSITIONS   = 10     # max concurrent positions
```
