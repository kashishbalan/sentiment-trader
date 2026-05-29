# Sentiment Alpha Trader

A live trading system that uses **FinBERT** to score financial news headlines
and generates long/short signals, executed via the **Alpaca** paper trading API.

---

## Architecture

```
NewsAPI headlines
      ↓
FinBERT (ProsusAI/finbert) sentiment scoring
      ↓
Signal generation (LONG / SHORT / FLAT)
      ↓
Alpaca paper trading execution (Market-on-Open orders)
      ↓
Dashboard monitoring
```

---

## Setup

### 1. Clone & install

```bash
git clone <repo>
cd sentiment_trader
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `NEWSAPI_KEY` — free at [newsapi.org](https://newsapi.org) (100 req/day free tier)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` — free paper account at [alpaca.markets](https://alpaca.markets)

### 3. Download FinBERT (first run only, ~500MB)

```python
from transformers import pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")
```

---

## Running

```bash
# Dry run — score tickers and print signals without trading
python main.py --mode dry-run

# Paper trade — execute on Alpaca paper account
python main.py --mode live

# Backtest — requires data/historical_sentiment.csv
python main.py --mode backtest
```

### Scheduled (run daily at 9:28 AM ET)

```bash
python scheduler.py
```

Or via cron:
```
28 9 * * 1-5 /path/to/venv/bin/python /path/to/sentiment_trader/main.py --mode live
```

---

## Generating Historical Sentiment (for backtest)

You'll need to build `data/historical_sentiment.csv` with columns:
`date, ticker, score, n_articles`

Options:
1. **Kaggle datasets** — search "financial news sentiment" for pre-labeled datasets
2. **GDELT** — large free news database with financial content
3. **Run scoring offline** — pull NewsAPI over time and save daily CSVs

Once you have it:
```bash
python main.py --mode backtest
```

---

## Tuning

Key parameters in `core/signals.py`:
```python
LONG_THRESHOLD  = 0.35   # increase to be more selective
SHORT_THRESHOLD = 0.35
MAX_POSITIONS   = 10
```

Run the backtest over different thresholds to find the optimal values before going live.

---

## Dashboard

Open `dashboard/index.html` in a browser for a live monitoring UI.
The dashboard simulates signals in demo mode; for live data, wire it to a
small Flask/FastAPI server that calls `main.py` and returns JSON.

---

## File Structure

```
sentiment_trader/
├── core/
│   ├── sentiment.py     # FinBERT scoring + NewsAPI fetching
│   ├── signals.py       # Signal generation + position sizing
│   ├── execution.py     # Alpaca order execution
│   └── backtest.py      # Historical backtest engine
├── data/                # CSVs saved here
├── dashboard/
│   └── index.html       # Monitoring dashboard
├── main.py              # Entry point
├── scheduler.py         # Daily scheduler
├── requirements.txt
└── .env.example
```

---

## Risk Management

- Equal dollar sizing across all positions
- Max 10 concurrent positions
- 5% cash buffer maintained
- Minimum 2 articles required before trading a ticker
- Hard stop-loss: implement in `execution.py` by checking open positions vs cost basis

---

## Disclaimer

This is for educational and paper trading purposes. Past simulated performance
does not guarantee future results. Do not risk money you cannot afford to lose.
