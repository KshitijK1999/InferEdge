# InferEdge — News-Flow Long/Short Signal

Daily equity long/short strategy built from news-flow data.

## Files

| File | Description |
|---|---|
| `InferEdge2.ipynb` | Interactive notebook — full pipeline from data load to results |
| `NEWS_SIGNAL_STRATEGY_WRITEUP.md` | Full writeup: construction, alternatives, results, caveats |
| `output/` | Generated results (diagnostics plot, daily CSV, summary JSON) |

## How to run

```bash
# 1. Place news_chunks.csv and prices.csv in a take_home_data/ folder
# 2. Open and run InferEdge2.ipynb
jupyter notebook InferEdge2.ipynb
```

## Approach summary

Ranks tickers each day by 3-day trailing chunk mention count (among covered names only),
goes long the top quintile and short the bottom quintile with equal dollar weights.
See `NEWS_SIGNAL_STRATEGY_WRITEUP.md` for the full methodology, data quality findings,
and limitations.
