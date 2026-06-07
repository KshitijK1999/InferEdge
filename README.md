# InferEdge — News-Flow Long/Short Signal

Daily equity long/short strategy built from news-flow data.

## Files

| File | Description |
|---|---|
| `news_ls_strategy.py` | Main pipeline — runs top-to-bottom, no hidden state |
| `InferEdge2.ipynb` | Interactive notebook mirroring the script |
| `NEWS_SIGNAL_STRATEGY_WRITEUP.md` | Full writeup: construction, alternatives, results, caveats |
| `output/` | Generated results (diagnostics plot, daily CSV, summary JSON) |

## How to run

```bash
# 1. Place news_chunks.csv and prices.csv in a take_home_data/ folder
# 2. Update NEWS_PATH / PRICES_PATH in news_ls_strategy.py if needed
python3 news_ls_strategy.py
```

## Approach summary

Ranks tickers each day by 3-day trailing chunk mention count (among covered names only),
goes long the top quintile and short the bottom quintile with equal dollar weights.
See `NEWS_SIGNAL_STRATEGY_WRITEUP.md` for the full methodology, data quality findings,
and limitations.
