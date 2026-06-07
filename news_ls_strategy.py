"""
Daily news-flow long/short signal: backtest and diagnostics.

Pipeline (linear, reproducible top-to-bottom — no hidden kernel state):
  1. Load news_chunks.csv and prices.csv, run basic structural EDA.
  2. Map every news timestamp to a trading day: stories before 9:00 count
     for that calendar day; stories at/after 9:00 roll to the next trading
     day. Weekend/holiday stories snap forward to the next valid trading day.
  3. De-duplicate news to the *story* level (chunks are sub-paragraphs of a
     story; counting them directly massively over-weights long-running
     stories — see EDA section).
  4. Build a daily per-ticker "story mention count", restricted to the priced
     universe, and smooth it with a short trailing window.
  5. Cross-sectionally rank tickers each day (among those with *any* recent
     coverage) and assign top/bottom-quintile long/short signals.
  6. Backtest: equal-dollar weights within each leg, dollar-neutral
     long/short, daily rebalance, close-to-close holding period.
  7. Report Sharpe and supporting diagnostics (turnover, IC, leg attribution,
     cost sensitivity).

Run with: python3 Backtest/news_ls_strategy.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Config / design choices (documented here so they're easy to find and change)
# ----------------------------------------------------------------------------
NEWS_PATH   = "take_home_data/news_chunks.csv"   # update to your local data path
PRICES_PATH = "take_home_data/prices.csv"        # update to your local data path
OUT_DIR     = "output"

NEWS_CUTOFF_HOUR = 9          # stories before 9 count for that day; 9+ roll to next trading day
MENTION_WINDOW_DAYS = 3       # trailing window over which mentions are summed
QUINTILE = 0.20               # top/bottom 20% -> long / short
ANNUALIZATION = 252
ROUND_TRIP_COST_BPS = 10      # illustrative cost: 10bps round-trip per name, per rebalance

import os
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
section("1. LOAD DATA")
news = pd.read_csv(NEWS_PATH)
news["ts"] = pd.to_datetime(news["ts"], format="ISO8601")
news["tickers"] = news["tickers"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
news = news[news["tickers"].apply(len) > 0].copy()

prices = pd.read_csv(PRICES_PATH)
prices["date"] = pd.to_datetime(prices["date"])

print(f"news_chunks: {len(news):,} rows | "
      f"{news['article_id'].nunique():,} articles | {news['story_id'].nunique():,} stories | "
      f"{news['ts'].min()} -> {news['ts'].max()}")
print(f"prices:      {len(prices):,} rows | {prices['ticker'].nunique():,} tickers | "
      f"{prices['date'].nunique()} trading days | {prices['date'].min().date()} -> {prices['date'].max().date()}")

# ----------------------------------------------------------------------------
# 2. EDA: why we de-duplicate to the story level, and why timing matters
# ----------------------------------------------------------------------------
section("2. STRUCTURAL EDA")

chunks_per_story = news.groupby("story_id").size()
print("Chunks per story_id:\n", chunks_per_story.describe()[["mean", "50%", "max"]])
biggest = chunks_per_story.sort_values(ascending=False).head(3)
print("\nLargest stories by chunk count (these are long-running, multi-month "
      "live/aggregated threads, NOT single news events):")
for sid, n in biggest.items():
    sub = news[news["story_id"] == sid]
    tix = sorted({t for row in sub["tickers"] for t in row})
    print(f"  {sid}: {n} chunks across {sub['ts'].dt.date.nunique()} distinct days "
          f"({sub['ts'].min().date()} -> {sub['ts'].max().date()}), tickers={tix[:6]}")
print("\n=> Counting raw chunks would let a handful of mega-threads dominate the "
      "mention ranking for whatever names they happen to tag. We instead count "
      "*distinct stories per ticker per day* — each story contributes at most once "
      "to a given ticker's daily count, however many chunks it was split into.")

n_tickers_per_chunk = news["tickers"].apply(len)
print(f"\nTickers per chunk: {(n_tickers_per_chunk == 0).mean():.1%} have none, "
      f"{(n_tickers_per_chunk == 1).mean():.1%} have exactly one "
      f"(US-ticker tagging is sparse and mostly single-name).")

print(f"\nNews timestamps span all 7 days of the week (markets are closed weekends/"
      f"holidays). Stories before {NEWS_CUTOFF_HOUR}:00 count toward that calendar day's "
      f"signal; stories at/after {NEWS_CUTOFF_HOUR}:00 roll to the next trading day. "
      f"Weekend/holiday stories snap forward to the next valid trading day.")

# ----------------------------------------------------------------------------
# 3. Map each news item to the trading day it contributes to.
#    Rule: hour < NEWS_CUTOFF_HOUR -> same calendar day; else -> next trading day.
#    searchsorted on the sorted trading-day list snaps weekends/holidays forward.
# ----------------------------------------------------------------------------
section("3. TIMESTAMP -> TRADING-DAY MAPPING")

trading_days = pd.DatetimeIndex(sorted(prices["date"].unique()))

news["_date"] = news["ts"].dt.normalize()
news["_hour"] = news["ts"].dt.hour
signal_date_raw = news["_date"] + pd.to_timedelta(
    (news["_hour"] >= NEWS_CUTOFF_HOUR).astype(int), unit="D"
)
news.drop(columns=["_date", "_hour"], inplace=True)

pos = np.searchsorted(trading_days.values, signal_date_raw.values, side="left")
in_range = pos < len(trading_days)
news = news[in_range].copy()
news["as_of_date"] = trading_days[pos[in_range]]

print(f"News rows mapped to a trading day: {len(news):,} "
      f"({len(news) / in_range.size:.1%} of raw rows; the rest postdate the last "
      f"trading day in prices.csv and are dropped).")

# ----------------------------------------------------------------------------
# 4. Build the daily per-ticker mention-count panel (story-deduplicated,
#    restricted to the priced universe), then smooth with a trailing window
# ----------------------------------------------------------------------------
section("4. SIGNAL CONSTRUCTION: TRAILING STORY-MENTION COUNT")

priced_tickers = set(prices["ticker"].unique())
exploded = news.explode("tickers").rename(columns={"tickers": "ticker"}).dropna(subset=["ticker"])
exploded = exploded[exploded["ticker"].isin(priced_tickers)]
print(f"Ticker mentions in news: {exploded['ticker'].nunique():,} unique symbols mentioned "
      f"(filtered to {len(priced_tickers):,} priced tickers).")

daily_counts = (exploded.groupby(["as_of_date", "ticker"]).size()
                .rename("n_chunks").reset_index())

news_dates = pd.DatetimeIndex(sorted(daily_counts["as_of_date"].unique()))
panel = (daily_counts.pivot(index="as_of_date", columns="ticker", values="n_chunks")
         .reindex(news_dates, fill_value=0)
         .fillna(0))

mention_signal = panel.rolling(MENTION_WINDOW_DAYS, min_periods=1).sum()
n_covered = (mention_signal > 0).sum(axis=1)


active_frac = n_covered / panel.shape[1]
print(f"\nWith a {MENTION_WINDOW_DAYS}-trading-day trailing window, on an average day "
      f"{n_covered.replace(0, np.nan).mean():.0f} of {panel.shape[1]} priced names "
      f"({active_frac[active_frac>0].mean():.0%}) have *any* recent coverage. The rest are "
      f"simply quiet — ranking them against each other would be ranking noise. We therefore "
      f"form quintiles only within the 'recently covered' subset each day (the key departure "
      f"from a naive full-universe percentile rank, which would fill the bottom quintile "
      f"almost entirely with arbitrary zero-mention names).")

# A second, important data-quality finding: there are stretches with literally
# zero captured stories — almost certainly feed outages, not "no news" (1,675
# liquid US names cannot generate zero stories for over a week in reality).
news_start_floor = news["as_of_date"].min()
zero_runs = (panel.sum(axis=1) == 0)
print("\nDays with literally ZERO captured stories across the whole universe "
      "(gap *starts* shown; runs of >=2 consecutive trading days after news coverage "
      "had already begun look like feed outages, not genuinely quiet markets):")
gap_run_lengths = zero_runs.groupby((~zero_runs).cumsum()).transform("sum")
outage_starts = panel.sum(axis=1)[(zero_runs) & (~zero_runs.shift(1, fill_value=False))
                                  & (gap_run_lengths >= 2)
                                  & (panel.index > pd.Timestamp("2026-02-23"))]
for d in outage_starts.index:
    run_len = gap_run_lengths.loc[d]
    print(f"  outage starting {d.date()}: {int(run_len)} consecutive trading days with zero stories")

# ----------------------------------------------------------------------------
# 5. Cross-sectional ranking -> long / short signal
# ----------------------------------------------------------------------------
covered  = mention_signal.where(mention_signal > 0)
pct_rank = covered.rank(axis=1, pct=True)

signal = pd.DataFrame(0, index=mention_signal.index, columns=mention_signal.columns)
signal[pct_rank >= 0.8] = 1
signal[pct_rank <= 0.2] = -1

n_long  = (signal == 1).sum(axis=1)
n_short = (signal == -1).sum(axis=1)
valid_day = pd.Series(True, index=mention_signal.index)  # all days included
print(f"Avg names per leg — long: {n_long[n_long>0].mean():.0f}, short: {n_short[n_short>0].mean():.0f}")

# ----------------------------------------------------------------------------
# 6. Backtest: capital-based iteration, hold until next rebalance date
# ----------------------------------------------------------------------------
section("6. BACKTEST")

signal         = signal.reindex(trading_days, fill_value=0)
mention_signal = mention_signal.reindex(trading_days, fill_value=0)

prices_pivot = prices.pivot(index="date", columns="ticker", values="close")

# Rebalance dates = signal dates that also have spot data.
# If a date has no spot data at all, it is dropped entirely.
all_signal_dates = signal.index[(signal != 0).any(axis=1)]
tradable_dates   = [d for d in all_signal_dates if d in prices_pivot.index]
print(f"Signal dates: {len(all_signal_dates)}  |  tradable (spot data present): {len(tradable_dates)}")

START_CAPITAL = 10_000
capital   = START_CAPITAL
records   = []
daily_nav = {}          # date -> portfolio NAV for every trading day (not just rebalances)
prev_longs  = set()
prev_shorts = set()

for i, d_entry in enumerate(tradable_dates[:-1]):
    d_exit = tradable_dates[i + 1]

    sigs         = signal.loc[d_entry]
    entry_prices = prices_pivot.loc[d_entry]
    exit_prices  = prices_pivot.loc[d_exit]

    # Drop tickers missing a price on either the entry or exit date
    longs  = [t for t in sigs[sigs ==  1].index
              if pd.notna(entry_prices.get(t)) and pd.notna(exit_prices.get(t))]
    shorts = [t for t in sigs[sigs == -1].index
              if pd.notna(entry_prices.get(t)) and pd.notna(exit_prices.get(t))]

    if not longs and not shorts:
        continue

    half_cap    = capital / 2
    long_alloc  = half_cap / len(longs)  if longs  else 0
    short_alloc = half_cap / len(shorts) if shorts else 0

    long_pnl  = sum(long_alloc  * (exit_prices[t] / entry_prices[t] - 1) for t in longs)
    short_pnl = sum(short_alloc * (1 - exit_prices[t] / entry_prices[t]) for t in shorts)

    period_ret = (long_pnl + short_pnl) / capital

    # Turnover: fraction of each leg that is NEW vs the previous rebalance
    long_to   = len(set(longs)  - prev_longs)  / len(longs)  if longs  else 0.0
    short_to  = len(set(shorts) - prev_shorts) / len(shorts) if shorts else 0.0
    cost_frac = (long_to + short_to) / 2 * ROUND_TRIP_COST_BPS / 10_000
    net_period_ret = period_ret - cost_frac
    prev_longs  = set(longs)
    prev_shorts = set(shorts)

    # Daily MTM for every trading day in this holding period [d_entry, d_exit)
    # Uses entry-date capital as base; prices for intermediate days valued mark-to-market
    for d_daily in [d for d in trading_days if d_entry <= d < d_exit]:
        if d_daily not in prices_pivot.index:
            daily_nav[d_daily] = capital
            continue
        dp = prices_pivot.loc[d_daily]
        d_long_pnl  = sum(long_alloc  * (dp[t] / entry_prices[t] - 1)
                          for t in longs  if pd.notna(dp.get(t)))
        d_short_pnl = sum(short_alloc * (1 - dp[t] / entry_prices[t])
                          for t in shorts if pd.notna(dp.get(t)))
        daily_nav[d_daily] = capital + d_long_pnl + d_short_pnl

    capital += long_pnl + short_pnl

    records.append({
        "entry_date":     d_entry,
        "exit_date":      d_exit,
        "n_long":         len(longs),
        "n_short":        len(shorts),
        "long_ret":       long_pnl  / half_cap if longs  else 0,
        "short_ret":      short_pnl / half_cap if shorts else 0,
        "period_ret":     period_ret,
        "net_period_ret": net_period_ret,
        "long_turnover":  long_to,
        "short_turnover": short_to,
        "capital":        capital,
    })

# Record the final day's NAV so the daily series is complete
if tradable_dates:
    daily_nav[tradable_dates[-1]] = capital

portfolio = pd.DataFrame(records).set_index("entry_date")
strat_ret = portfolio["period_ret"]
net_ret   = portfolio["net_period_ret"]

# Daily NAV series — basis for drawdown and Sharpe computed on actual daily returns
nav_series  = pd.Series(daily_nav).sort_index()
cum_daily   = nav_series / nav_series.iloc[0]
dd_daily    = cum_daily / cum_daily.cummax() - 1
daily_ret   = nav_series.pct_change().dropna()

# Period-level metrics (rebalance frequency returns)
mean_d   = strat_ret.mean()
std_d    = strat_ret.std()
sharpe   = mean_d / std_d * np.sqrt(ANNUALIZATION)
cum      = (1 + strat_ret).cumprod()
dd       = cum / cum.cummax() - 1
n_years  = (portfolio["exit_date"].max() - portfolio.index.min()).days / 365.25
cagr     = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
ann_ret  = (1 + mean_d) ** ANNUALIZATION - 1
hit_rate = (strat_ret > 0).mean()

# Net metrics (after turnover-based round-trip cost)
net_sharpe = net_ret.mean() / net_ret.std() * np.sqrt(ANNUALIZATION) if net_ret.std() > 0 else np.nan
net_ann    = (1 + net_ret.mean()) ** ANNUALIZATION - 1

# Longest drawdown duration (in rebalance periods)
in_dd   = (dd < 0).astype(int)
groups  = (in_dd != in_dd.shift()).cumsum()
longest_dd_periods = int(in_dd.groupby(groups).sum().max())

print(f"\nStart capital: ${START_CAPITAL:,.0f}  ->  End capital: ${capital:,.2f}")
print(f"Backtest window:         {portfolio.index.min().date()} -> "
      f"{portfolio['exit_date'].max().date()} ({len(portfolio)} periods)")
print(f"Total return:            {cum.iloc[-1] - 1:.2%}")
print(f"CAGR:                    {cagr:.2%}")
print(f"Annualized volatility:   {std_d * np.sqrt(ANNUALIZATION):.2%}")
print(f"Sharpe ratio (gross):    {sharpe:.2f}")
print(f"Net Sharpe ({ROUND_TRIP_COST_BPS}bps cost):  {net_sharpe:.2f}")
print(f"Net annualized return:   {net_ann:.2%}")
print(f"Hit rate (periods > 0):  {hit_rate:.1%}")
print(f"Max drawdown:            {dd_daily.min():.2%}")
print(f"Longest drawdown:        {longest_dd_periods} periods")
print(f"Avg n_long / n_short:    {portfolio['n_long'].mean():.0f} / {portfolio['n_short'].mean():.0f}")

# ----------------------------------------------------------------------------
# 7. Plots
# ----------------------------------------------------------------------------
section("7. PLOTS")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
cum_daily.plot(ax=ax, label="Long/Short (daily NAV)", lw=1.6)
((1 + portfolio["long_ret"]).cumprod()).plot(ax=ax, label="Long leg", lw=1, alpha=0.7)
((1 + portfolio["short_ret"]).cumprod()).plot(ax=ax, label="Short leg P&L", lw=1, alpha=0.7)
ax.set_title("Cumulative return (daily NAV)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
dd_daily.plot(ax=ax, color="firebrick")
ax.fill_between(dd_daily.index, dd_daily.values, 0, color="firebrick", alpha=0.2)
ax.set_title("Drawdown (daily NAV)"); ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(strat_ret * 100, bins=20, edgecolor="black", alpha=0.75)
ax.axvline(strat_ret.mean() * 100, color="red", ls="--", label=f"mean={strat_ret.mean():.3%}")
ax.set_title("Period L/S return distribution (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = f"{OUT_DIR}/news_ls_diagnostics.png"
plt.savefig(fig_path, dpi=140)
print(f"Saved diagnostic plots -> {fig_path}")

# ----------------------------------------------------------------------------
# 8. Persist results
# ----------------------------------------------------------------------------
section("8. SAVE OUTPUTS")
results = portfolio.copy()
results["cum_strategy"] = cum
results.to_csv(f"{OUT_DIR}/daily_results.csv")
print(f"Saved results -> {OUT_DIR}/daily_results.csv")

summary = {
    "window_start":       str(portfolio.index.min().date()),
    "window_end":         str(portfolio["exit_date"].max().date()),
    "n_periods":          int(len(strat_ret)),
    "sharpe_gross":       float(sharpe),
    "sharpe_net":         float(net_sharpe),
    "cagr":               float(cagr),
    "ann_return":         float(ann_ret),
    "net_ann_return":     float(net_ann),
    "ann_vol":            float(std_d * np.sqrt(ANNUALIZATION)),
    "max_drawdown":       float(dd_daily.min()),
    "longest_dd_periods": int(longest_dd_periods),
    "hit_rate":           float(hit_rate),
    "avg_n_long":         float(portfolio["n_long"].mean()),
    "avg_n_short":        float(portfolio["n_short"].mean()),
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary -> {OUT_DIR}/summary.json")
print("\nSUMMARY:", json.dumps(summary, indent=2))
