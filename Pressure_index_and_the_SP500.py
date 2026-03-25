"""
Inspired by this FT article 
and the Deutsche Bank Chart: https://www.ft.com/content/16c122a3-d2f3-4ae5-8c98-af1770d99e0f?syn-25a6b1a6=1

Hypothesis: Sustained weakness across broad macroeconomic indicators 
should push the US government and Fed to enact a response 
that will support the broad equity market. 

Creating a Market Pressure Index
======================
Composite daily index = equally weighted average of rolling 30-day z-scores 
of the 20-day change in:
    
  1. S&P 500                        (FRED: SP500)           — 20%   
  2. 10Y Treasury yield              (FRED: DGS10)           — 20%
  3. 5y5y inflation fwd              (FRED: T5YIFR)           — 20%
  4. US Economic Policy Uncertainty  (FRED: USEPUINDXD, 7dMA) — 20%
  5. BBB corporate credit spread     (FRED: BAMLC0A4CBBB)    — 20%

Convention: a POSITIVE z-score means conditions are improving
(higher S&P, lower yields, lower inflation expectations, 
 lower policy uncertainty, tighter credit spreads). We can debate
on the signs for yield and inflation expectations, but for the sake 
of this simple exercise, we stick with the above.

Requirements:
    pip install pandas matplotlib requests
"""

import datetime as dt
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import requests
from io import StringIO

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1. Parameters
# ──────────────────────────────────────────────────────────────
WINDOW_CHANGE = 20      # 20-business-day change
WINDOW_ZSCORE = 30      # 30-day rolling z-score window *roughly 1.5 months
WINDOW_EPU_MA = 7       # 7-day moving average to smooth daily EPU
START_FETCH   = "2016-01-01"   # fetch extra history for rolling windows
END_FETCH     = dt.date.today().isoformat()
INDEX_START   = "2016-01-01"   # composite index starts here, though the S&P500 series on FRED only starts from 28 March 2016

# ──────────────────────────────────────────────────────────────
# 2. Helper: fetch a FRED series as a DataFrame via CSV endpoint
# ──────────────────────────────────────────────────────────────
def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Download a single FRED series via the public CSV endpoint."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    s = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True,
                     na_values=".")
    return s.iloc[:, 0].dropna().astype(float)

# ──────────────────────────────────────────────────────────────
# 3. Fetch all FRED series
# ──────────────────────────────────────────────────────────────
print("Fetching FRED data …")
sp500  = fetch_fred("SP500",     START_FETCH, END_FETCH)
dgs10  = fetch_fred("DGS10",      START_FETCH, END_FETCH)
t5yifr = fetch_fred("T5YIFR",     START_FETCH, END_FETCH)
epu    = fetch_fred("USEPUINDXD",   START_FETCH, END_FETCH)
bbb    = fetch_fred("BAMLC0A4CBBB", START_FETCH, END_FETCH)
print(f"  SP500       : {sp500.index.min().date()} → {sp500.index.max().date()}  ({len(sp500)} obs)")
print(f"  DGS10       : {dgs10.index.min().date()} → {dgs10.index.max().date()}  ({len(dgs10)} obs)")
print(f"  T5YIFR      : {t5yifr.index.min().date()} → {t5yifr.index.max().date()}  ({len(t5yifr)} obs)")
print(f"  USEPUINDXD  : {epu.index.min().date()} → {epu.index.max().date()}  ({len(epu)} obs)")
print(f"  BAMLC0A4CBBB: {bbb.index.min().date()} → {bbb.index.max().date()}  ({len(bbb)} obs)")

# ──────────────────────────────────────────────────────────────
# 4. Build a common daily calendar (business days from S&P)
# ──────────────────────────────────────────────────────────────
df = pd.DataFrame(index=sp500.index)
df["SP500"]  = sp500
df["DGS10"]  = dgs10.reindex(df.index)
df["T5YIFR"] = t5yifr.reindex(df.index)
df["BBB"]    = bbb.reindex(df.index)

# EPU is daily (incl. weekends); reindex to business days, then smooth
df["EPU_raw"] = epu.reindex(df.index, method="ffill")
df["EPU"]     = df["EPU_raw"].rolling(WINDOW_EPU_MA, min_periods=1).mean()

# Forward-fill FRED gaps (holidays) then drop any remaining NaN
df = df.ffill().dropna()

# ──────────────────────────────────────────────────────────────
# 5. Compute 20-day changes
# ──────────────────────────────────────────────────────────────
# For S&P 500: percent change
df["chg_SP500"] = df["SP500"].pct_change(WINDOW_CHANGE) * 100

# For yields & inflation expectations: arithmetic change in percentage points
df["chg_DGS10"]  = df["DGS10"].diff(WINDOW_CHANGE)
df["chg_T5YIFR"] = df["T5YIFR"].diff(WINDOW_CHANGE)

# For EPU (7-day MA smoothed): percentage change
df["chg_EPU"] = df["EPU"].pct_change(WINDOW_CHANGE) * 100

# For BBB spread: arithmetic change in percentage points
df["chg_BBB"] = df["BBB"].diff(WINDOW_CHANGE)

# ──────────────────────────────────────────────────────────────
# 6. Rolling 30-day z-scores
# ──────────────────────────────────────────────────────────────
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mu  = series.rolling(window, min_periods=window).mean()
    sig = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mu) / sig

df["z_SP500"]  =  rolling_zscore(df["chg_SP500"],  WINDOW_ZSCORE)
# SIGN FLIP: rising yields / inflation / uncertainty / spreads = stress → negative z
df["z_DGS10"]  = -rolling_zscore(df["chg_DGS10"],  WINDOW_ZSCORE)
df["z_T5YIFR"] = -rolling_zscore(df["chg_T5YIFR"], WINDOW_ZSCORE)
df["z_EPU"]    = -rolling_zscore(df["chg_EPU"],     WINDOW_ZSCORE)
df["z_BBB"]    = -rolling_zscore(df["chg_BBB"],     WINDOW_ZSCORE)

z_cols = ["z_SP500", "z_DGS10", "z_T5YIFR", "z_EPU", "z_BBB"]

# ──────────────────────────────────────────────────────────────
# 7. Composite index = weighted average of z-scores
# ──────────────────────────────────────────────────────────────
# Equal weight: 20% each (totals 100%)
z_weights = {"z_SP500": 0.20, "z_DGS10": 0.20, "z_T5YIFR": 0.20,
             "z_EPU":   0.20, "z_BBB":   0.20}
df["composite"] = sum(df[col] * w for col, w in z_weights.items())

# Trim to display window
idx = df.loc[INDEX_START:].copy()
print(f"\nComposite index: {idx.index.min().date()} → {idx.index.max().date()}")
print(f"Latest value:    {idx['composite'].iloc[-1]:.3f}")
print(f"Min / Max:       {idx['composite'].min():.3f} / {idx['composite'].max():.3f}")

# ──────────────────────────────────────────────────────────────
# 8. Export to CSV
# ──────────────────────────────────────────────────────────────
out_cols = (
    ["SP500", "DGS10", "T5YIFR", "EPU", "BBB"]
    + [f"chg_{c}" for c in ["SP500", "DGS10", "T5YIFR", "EPU", "BBB"]]
    + z_cols
    + ["composite"]
)
idx[out_cols].to_csv("pressure_index.csv")
print("Saved → pressure_index.csv")

# ──────────────────────────────────────────────────────────────
# 9. Plot 1 — Composite pressure index (single panel)
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(idx.index, 0, idx["composite"],
                where=idx["composite"] >= 0, color="#2ca02c", alpha=0.25)
ax.fill_between(idx.index, 0, idx["composite"],
                where=idx["composite"] <  0, color="#d62728", alpha=0.25)
ax.plot(idx.index, idx["composite"], color="#1f3b73", lw=1.8)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_ylabel("Composite z-score", fontsize=12)
ax.set_title("Market Pressure Index\n"
             "(equal-weighted z-scores: S&P 500, 10Y yield, 5y5y inflation, EPU, BBB spread)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.autofmt_xdate(rotation=0, ha="center")
fig.text(0.01, 0.01, "Source: FRED", fontsize=9, color="grey",
         ha="left", va="bottom")
fig.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("pressure_index.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → pressure_index.png")

# ──────────────────────────────────────────────────────────────
# 10. Identify pressure episodes (composite < -1.0 for 5 consec. days)
# ──────────────────────────────────────────────────────────────
THRESHOLD    = -1.0
CONSEC_DAYS  = 5
FORWARD_DAYS = 20   # trading days to track performance

# Work on the trimmed index (idx) which starts at INDEX_START
below = (idx["composite"] < THRESHOLD).astype(int)

# Rolling sum of the flag — equals CONSEC_DAYS when we have 5 in a row
streak = below.rolling(CONSEC_DAYS, min_periods=CONSEC_DAYS).sum()

# The signal fires on the 5th consecutive day below threshold
signal = streak == CONSEC_DAYS

# Group consecutive signal days into episodes: only keep the *first*
# day of each cluster (i.e. the day the 5-day condition is first met)
signal_dates = idx.index[signal]
episode_starts = []
for d in signal_dates:
    if not episode_starts or (d - episode_starts[-1]).days > 10:
        episode_starts.append(d)

print(f"\nPressure episodes (composite < {THRESHOLD} for {CONSEC_DAYS}+ days):")
for i, d in enumerate(episode_starts):
    val = idx.loc[d, "composite"]
    print(f"  Episode {i+1}: {d.date()}  (composite = {val:.2f})")

# ──────────────────────────────────────────────────────────────
# 11. Build forward return paths
# ──────────────────────────────────────────────────────────────
# Use the full df (not trimmed idx) for forward S&P prices
all_dates = df.index.tolist()

paths = {}
for i, start in enumerate(episode_starts):
    start_loc = all_dates.index(start)
    # Grab FORWARD_DAYS business days *after* the trigger (day 0 = trigger)
    end_loc = min(start_loc + FORWARD_DAYS, len(all_dates) - 1)
    window = df.iloc[start_loc : end_loc + 1]
    # Cumulative return rebased to 0% on day 0
    cum_ret = (window["SP500"] / window["SP500"].iloc[0] - 1) * 100
    cum_ret.index = range(len(cum_ret))  # day 0, 1, 2, …
    paths[f"Ep {i+1}: {start.strftime('%d-%b-%Y')}"] = cum_ret

returns_df = pd.DataFrame(paths)
print(f"\nForward return paths built ({len(returns_df)} days max)")

# ──────────────────────────────────────────────────────────────
# 12. Plot 2 — Episode chart (auto y-axis)
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

cmap = plt.cm.tab10
for i, col in enumerate(returns_df.columns):
    color = cmap(i % 10)
    ax.plot(returns_df.index, returns_df[col],
            lw=2, color=color, label=col, alpha=0.85)
    last_idx = returns_df[col].dropna().index[-1]
    last_val = returns_df[col].dropna().iloc[-1]
    ax.scatter(last_idx, last_val, color=color, s=40, zorder=5)

ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Business days since trigger (day 0 = 5th consec. day < -1.0)",
              fontsize=11)
ax.set_ylabel("S&P 500 cumulative return (%)", fontsize=12)
ax.set_title("S&P 500 Performance After Pressure Index Episodes\n"
             f"(triggered when composite z-score < {THRESHOLD} "
             f"for {CONSEC_DAYS} consecutive days)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, FORWARD_DAYS + 0.5)
fig.text(0.01, 0.01, "Source: FRED", fontsize=9, color="grey",
         ha="left", va="bottom")
fig.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("sp500_after_pressure.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → sp500_after_pressure.png")

# ──────────────────────────────────────────────────────────────
# 13. Plot 3 — Zoomed-in Episode chart (fixed y-axis -5% to 10%)
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

for i, col in enumerate(returns_df.columns):
    color = cmap(i % 10)
    ax.plot(returns_df.index, returns_df[col],
            lw=2, color=color, label=col, alpha=0.85)
    last_idx = returns_df[col].dropna().index[-1]
    last_val = returns_df[col].dropna().iloc[-1]
    ax.scatter(last_idx, last_val, color=color, s=40, zorder=5)

ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Business days since trigger (day 0 = 5th consec. day < -1.0)",
              fontsize=11)
ax.set_ylabel("S&P 500 cumulative return (%)", fontsize=12)
ax.set_title("S&P 500 Performance After Pressure Index Episodes\n"
             f"(triggered when composite z-score < {THRESHOLD} "
             f"for {CONSEC_DAYS} consecutive days)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.set_ylim(-5, 10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, FORWARD_DAYS + 0.5)
fig.text(0.01, 0.01, "Source: FRED", fontsize=9, color="grey",
         ha="left", va="bottom")
fig.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("sp500_after_pressure_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → sp500_after_pressure_zoom.png")

# Also save the return paths to CSV
returns_df.to_csv("sp500_after_pressure.csv", index_label="day")
print("Saved → sp500_after_pressure.csv")
