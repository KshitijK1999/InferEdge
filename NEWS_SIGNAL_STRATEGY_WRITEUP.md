# News-Flow Long/Short Signal — Strategy Writeup

Code: `Backtest/news_ls_strategy.py` (linear, reproducible top-to-bottom).

---

## 1. Signal Construction

**Input data.** The news dataset has 592,510 rows, but those rows are *chunks* — sub-paragraphs of 29,845 distinct stories (which in turn come from 97,544 articles). The data model matters: a single story is split into many chunks, so the hierarchy is article → story → N chunks. Each chunk carries a UTC timestamp and a list of associated tickers.

**Timing — no look-ahead.** Stories arrive around the clock and on weekends. I map each chunk's timestamp to the trading day it should inform using one simple rule: a timestamp before 09:00 contributes to that calendar day's signal; at or after 09:00 it rolls to the next trading day. Weekend and holiday timestamps snap forward to the next valid trading day via binary search on the sorted trading-day index (`np.searchsorted`). This is the critical design choice that prevents tomorrow's news from entering today's portfolio.

**Universe filter.** Only tickers present in the prices file (1,675 names) are included. All 1,675 priced names had at least some news coverage — no ticker-mapping problem.

**Mention signal.** I count raw chunks per (ticker, day), then sum over a trailing 3-day window (`mention_signal`). The 3-day window is a deliberate tradeoff: a single day is too sparse (most names appear in 0–2 chunks; ties dominate any rank), while a longer window stales the signal and burns more of an already-short sample.

**Cross-sectional ranking.** On an average day, roughly 46% of the 1,675 names have any recent coverage. Ranking zero-mention names against each other is ranking noise — so quintiles are formed only within the "covered" subset (mention_signal > 0 on that day). Top 20% → long (+1), bottom 20% → short (−1). Equal-dollar weights within each leg, dollar-neutral (50% long / 50% short).

**Backtest.** Capital-based iteration starting at $10,000: enter at the signal date's close, hold until the next rebalance date's close, drop any ticker missing a price on either the entry or exit date.

---

## 2. Data Investigation — Key Findings

**Chunk inflation via mega-threads.** The largest single story has 1,762 chunks across 57 trading days — it is not a news event, it is a live-updating aggregate thread that continuously re-tags the same tickers. A handful of such threads would completely dominate a raw-chunk mention rank. I investigated two responses: (a) de-duplicate to one count per (story, ticker, day), or (b) count chunks anyway and accept the inflation. I chose chunk counting to capture genuine repetition of recent stories, accepting the mega-thread risk as a known caveat.

**Feed outages.** Three multi-day stretches show zero chunks across all 1,675 names: Feb 24–27, Mar 16–19, and Apr 13–22 (8 consecutive trading days). 1,675 liquid US names cannot generate zero news for 1.5 weeks — this is a data collection gap, not a quiet market. With a 3-day rolling window and `min_periods=1`, pre-outage signal bleeds into the outage period; those dates carry stale information dressed as fresh signal.

**Ticker sparsity.** 99%+ of chunks tag exactly one ticker. US ticker attribution in the feed is sparse and mostly single-name — there is no meaningful co-occurrence signal to exploit.

**The size-factor diagnostic.** To test whether the raw mention-count signal is genuine news alpha or just a proxy for market cap (large-caps are structurally always in the news), I ran a second version: z-score each ticker's 3-day mention count against its own trailing 20-day mean and standard deviation. This "abnormal coverage" signal should isolate *unusual* attention rather than rewarding coverage level. The result: **the signal inverted** (gross Sharpe −2.08 vs +2.38 for raw counts). This is direct evidence that the raw mention-count signal is largely a size/coverage-factor bet — "go long the names that are always in the news" happened to work in this window but is not grounded in news content.

---

## 3. Results (raw mention-count signal)

| Metric | Value |
|---|---|
| Window | 2026-02-23 → 2026-05-15 (41 periods) |
| Gross Sharpe | **2.38** |
| Net Sharpe (10 bps round-trip) | **0.61** |
| CAGR | 10.4% |
| Hit rate | 58.5% |
| Max drawdown (daily NAV) | −10.38% |
| Avg turnover per rebalance — long / short | 31.7% / 49.6% |
| Cross-sectional IC (mean / t-stat) | −0.004 / −0.52 |

One methodological note on the drawdown: measuring it only at rebalance boundaries — as most simple backtest loops do — shows −1.40%. Tracking actual daily NAV between rebalances (positions marked to market every trading day) reveals the true figure is **−10.38%**. The strategy carries meaningful intraday risk that the period-level view hides entirely.

The IC of −0.004 (t ≈ 0.5) is indistinguishable from zero. The gross Sharpe of 2.38 is driven by hit rate over a tiny sample, not by a statistically significant information edge. At 10 bps round-trip cost, the 31–50% per-leg turnover reduces net Sharpe to 0.61.

---

## 4. Alternatives Considered

| Choice | What I tried | Why I landed where I did |
|---|---|---|
| Chunk vs story counting | Both | Chose chunks; mega-thread inflation is a known caveat |
| Raw count vs z-score (abnormal) | Both | Z-score inverted the signal, revealing raw count is a size proxy |
| Timezone conversion | UTC→ET vs simple hour cutoff | Simple hour cutoff is transparent and avoids DST edge cases; UTC→ET adds complexity with no clear benefit given the data quality issues |
| Universe filter | All tickers vs priced-only | Priced-only is the correct constraint; untradable names add noise |
| Min-covered gate | Tried removing it | No gate; all days included with valid signal |
| Full-universe rank vs covered-only | Both | Covered-only avoids ranking silence against silence |

---

## 5. What I'd Do with More Time

Grouped by what kind of improvement each addresses.

### 5a. Signal quality

**Sentiment / direction.** The single biggest gap: all coverage scores the same regardless of whether it is an earnings miss, a product launch, or routine analyst chatter. Even crude lexicon-based sentiment (VADER, FinBERT) on chunk text would turn this from an attention proxy into a directional signal — and the z-score experiment (§3) suggests direction is where any real edge lives.

**Abnormal silence as a contrarian signal.** The current signal only acts on tickers that are covered. A ticker that has been in the news every day for two weeks and suddenly goes quiet might be equally interesting — the absence of coverage after sustained attention is itself information.

**Story freshness / exponential decay.** The 3-day window treats yesterday's story and a 3-day-old story identically. Exponential decay (half-life ~1 day) would weight breaking news more heavily and stale coverage less, which better matches how markets process information.

**Story type.** Not all news is equal: earnings announcements, M&A filings, analyst initiations, and routine press releases carry very different information content. If the article or story metadata includes a category or source type, conditioning the signal on story type would substantially improve precision.

**Multi-frequency combination.** A single 3-day window forces a single time horizon. Combining signals at 1d, 3d, and 5d (with appropriate weights, e.g. via PCA or simple averaging after z-scoring each) would hedge against the signal's sensitivity to window choice and likely improve IC.

**Named entity / ticker quality.** The feed's ticker tagging is sparse and sometimes noisy. Better entity disambiguation — confirming a chunk mentioning "Apple" is tagging AAPL vs a downstream supplier — would reduce false positives in the mention count.

### 5b. Portfolio construction

**Buffer zone to reduce turnover.** The single most direct lever on net returns: exit a long only when rank falls below the 30th percentile (not the 80th entry threshold). This could cut 31–50% per-leg turnover to ~15% with minimal signal loss, likely pushing net Sharpe from 0.61 toward 1.5+.

**Sector-neutral ranking.** Running percentile ranks within sector buckets isolates stock-level news alpha from sector rotation. A naive full-universe rank fills the long book with whichever sector dominated the headlines that day — a sector bet, not a news-flow bet.

**Factor neutralization.** Regress the mention signal against known factors (market cap, momentum, value) and act on the residual. This removes the size-proxy effect we directly observed: the raw signal's Sharpe of +2.38 inverted to −2.08 once we z-scored within each ticker's own history, which is consistent with the raw version being a disguised size/coverage factor.

**Volatility-scaled position sizing.** Equal-dollar weights ignore that a $5,000 position in a 60% annualised-vol small-cap and the same position in a 15% vol large-cap carry very different risk. Sizing inversely proportional to 20-day realised volatility (risk parity within each leg) would give a more stable risk contribution per name.

**Liquidity filter.** ~170 names per leg, rebalanced daily, almost certainly includes illiquid mid/small-caps where close-to-close fills are unrealistic. Restricting to names above a minimum average daily volume (e.g., $10M ADV) reduces the universe but makes the cost assumptions defensible.

### 5c. Statistical rigour

**Walk-forward validation.** With a static in-sample backtest over 41 periods, there is no way to distinguish a real edge from noise. A walk-forward setup — train the window choice and quintile threshold on one period, test on the next, roll forward — would give an honest out-of-sample performance estimate.

**Bootstrap confidence intervals on Sharpe.** The point estimate of 2.38 has enormous uncertainty over 41 observations. Bootstrapping the period-return series gives a proper confidence interval (likely spanning 0 to 4+), making clear how little we can conclude from this sample size alone.

**IC decay curve.** Compute IC at lag 1, 2, 3, … days to see how fast the signal decays. If IC at lag 1 is near-zero but IC at lag 0 (same-day) is positive, the issue is entry timing. If IC at lag 3 is still as large as lag 1, a longer holding period is justified. This diagnostic directly informs holding-period and window-length choices.

**Separating outage periods.** The three feed outage stretches corrupt the rolling signal. Re-running the backtest excluding those periods explicitly (and the 2–3 days after each outage where the window carries stale data) would give a cleaner picture of whether the signal works on clean data.

### 5d. Live system design

**T+1 open entry.** Positions are currently opened at the same close used to form the signal. The earliest realistic entry is T+1 open; this one-day lag would modestly reduce the Sharpe but is necessary for any real implementation.

**Intraday signal refresh.** The current signal is daily, but news arrives continuously. A live system could update mention counts every hour during the trading day and act on material moves in rank without waiting for the next daily rebalance — reducing entry lag without requiring intraday price data.

**Short book borrow cost.** The 10 bps round-trip cost in the net Sharpe ignores stock borrow. Liquid large-caps typically borrow at 20–50 bps annually; names that appear in the short book because of unusual negative attention (exactly the names most likely to be hard to borrow) can cost 200–500 bps or more. A realistic net Sharpe must include estimated borrow.

**Outage / data-quality detection.** The three feed gaps in this dataset were only visible in hindsight. A live system needs an automated signal-health check: if fewer than X names have been tagged in the last 4 hours, flag the signal as stale and hold or flatten positions rather than acting on corrupted data.

---

## 6. Limitations & Caveats

- **No sentiment.** This is not a news-quality signal — it is a news-quantity signal. We are betting on attention, not information content. The z-score experiment strongly suggests the raw-count version is primarily a size-factor bet in disguise.
- **Sample size.** 41 periods. IC t-stat of 0.5. Any headline Sharpe should be treated as illustrative; there is no statistical basis to claim a real edge from this window alone.
- **Feed outages corrupt ~20% of the sample.** Three multi-day gaps mean the signal on dates immediately following each outage is stale pre-outage data, not fresh information.
- **Mega-thread inflation.** A handful of long-running aggregate threads inflate chunk counts for certain names; the signal partly reflects the tagging density of the feed, not genuine news intensity.
- **Entry timing.** Positions are opened at the same close used to form the signal. In practice the earliest realistic entry is T+1 open.
- **Equal weighting ignores liquidity.** ~170 names per leg almost certainly includes names where a close-to-close fill assumption is unrealistic.
- **Drawdown is understated at rebalance granularity.** The commonly-cited −1.40% max drawdown hides the real −10.38% figure visible only in daily NAV.
- **Static universe.** No entries, exits, or delistings across the window; a live system would face universe churn not captured here.
