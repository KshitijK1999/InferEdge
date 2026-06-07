# News-Flow Long/Short Signal — Strategy Writeup

---

## Input Data

The news dataset has 592,510 rows, but those rows are *chunks* — sub-paragraphs of 29,845 distinct stories (which in turn come from 97,544 articles). A single story is split into many chunks, so the hierarchy is **article → story → N chunks**, where each story has a timestamp and a list of associated tickers. Additionally we have spot data (OHLC) across all tickers.

---

## Data Filtering

Pre-processing steps included:

- **Ticker parsing:** Parsed JSON-encoded tickers column; dropped rows that failed to parse or contained no tickers.
- **Post-date drop:** Chunks mapping beyond the last date in `prices.csv` dropped.
- **Null-ticker drop:** Rows with null or empty ticker values removed.
- **Per-ticker price check:** Tickers missing entry or exit close prices dropped from that period's portfolio.

**Feed outages corrupt ~20% of the sample.** Three multi-day gaps mean the signal on dates immediately following each outage is stale pre-outage data, not fresh information.

**Mega-thread inflation.** A handful of long-running aggregate threads inflate chunk counts for certain names; the signal partly reflects the tagging density of the feed, not genuine news intensity.

---

## Signal Generation

The news data arrives as raw chunks around the clock and on weekends. Each chunk's timestamp is mapped to the trading day it should inform using one simple rule: a timestamp before 09:00 contributes to that calendar day's signal; at or after 09:00 it rolls to the next trading day. Weekend and holiday timestamps snap forward to the next valid trading day. This ensures no forward-looking bias enters the portfolio.

Once mapped to trading days, the data is filtered to the 1,675 tickers with available price data and exploded so each row represents a single (chunk, ticker, day) triple. These are aggregated into a daily mention count per ticker, then smoothed with a **3-day trailing sum** — a deliberate tradeoff between noise reduction (a single day is too sparse; most names appear in 0–2 chunks and ties dominate any rank) and signal freshness (longer windows stale the information).

The resulting mention counts are ranked cross-sectionally each day, but only within the subset of tickers that have any recent coverage. Tickers in the top 20% by mention count are assigned a long signal (+1) and the bottom 20% a short signal (−1). Equal dollar weights within each leg, dollar-neutral overall.

---

## Assumptions

- **All news is good news.** We have no sentiment labels, so we treat mention count as a positive proxy. The signal implicitly assumes more coverage → more positive attention → price appreciation.
- **Silence carries no information.** We rank only among covered names and ignore the uncovered universe entirely. This assumes zero mentions is just "no data" rather than a signal in itself.

---

## Alternatives Considered

- **Chunk vs. story counting:** Tested both aggregation units. Chunks were preferred because story-level aggregation loses the intensity signal carried by how much text a topic generates, though mega-thread inflation remains a known caveat.
- **Raw count vs. z-score (abnormal mention rate):** Tested both. Z-scoring inverted the signal (Sharpe −2.08 vs +2.38), suggesting raw count is acting as a size proxy — larger, more liquid names attracting more coverage — and that size tilt appears to be the return driver rather than abnormal attention.

---

## Further Improvements

- **Sentiment / direction.** Even crude sentiment on chunk text would turn this from an attention proxy into a directional signal. The z-score experiment suggests direction is where any real edge lives.
- **Story freshness / exponential decay.** Exponential decay would weight breaking news more heavily and stale coverage less, which better matches how markets process information.
- **Buffer zone to reduce turnover.** The single most direct lever on net returns: exit a long only when rank falls below the 30th percentile (not the 80th entry threshold), cutting 31–50% per-leg turnover to ~15% with minimal signal loss.
- **Sector-neutral ranking.** Running percentile ranks within sector buckets isolates stock-level news alpha from sector rotation.
- **Structured backtester architecture.** The current implementation is a monolithic top-to-bottom script where signal generation, trade execution, and state tracking are all interleaved. A cleaner design separates these into three layers: a `TradingStrategy` class that owns only the signal-to-weight mapping, an `InventoryStatus` class for managing positions, and a `BacktestRunner` that orchestrates the loop without holding any state itself.
