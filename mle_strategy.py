"""
Macro-Liquidity-EPS (MLE) DAX Switching Strategy
==================================================

Switches between 0% and 100% DAX allocation using a composite signal
built from three independent pillars:

  PILLAR 1 — MACRO  : Copper/Gold ratio (global economic cycle proxy)
  PILLAR 2 — LIQUIDITY : VIX z-score + EUR/USD trend + HY credit spreads
  PILLAR 3 — EPS    : Cyclicals/Defensives ratio + rate-change drag

Each pillar is scored –1 / 0 / +1.  Composite = sum (range –3 to +3).
→ Long DAX (100%) when composite ≥ +1  (majority bullish)
→ Cash      (  0%) when composite <  +1
Position is set with a 1-month lag to avoid look-ahead bias.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "dax":    "^GDAXI",
    "copper": "HG=F",
    "gold":   "GC=F",
    "eurusd": "EURUSD=X",
    "vix":    "^VIX",
    "hyg":    "HYG",   # High-Yield Bond ETF (credit/liquidity)
    "lqd":    "LQD",   # Investment-Grade Bond ETF
    "xly":    "XLY",   # Consumer Discretionary  (cyclicals)
    "xlp":    "XLP",   # Consumer Staples         (defensives)
    "tnx":    "^TNX",  # US 10-year yield (rate-change drag on EPS)
}

START  = "2005-01-01"
END    = "2024-12-31"
REBAL  = "ME"          # month-end rebalancing
THRESHOLD = 1          # composite >= this → long DAX

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    print("  Downloading price data via yfinance …")
    raw = yf.download(
        list(TICKERS.values()), start=START, end=END,
        auto_adjust=True, progress=False
    )["Close"]
    # yfinance returns columns sorted alphabetically — rename via reverse mapping
    raw = raw.rename(columns={v: k for k, v in TICKERS.items()})
    # Resample to month-end, forward-fill at most 5 days for missing quotes
    monthly = raw.resample(REBAL).last().ffill(limit=5)
    monthly.dropna(how="all", inplace=True)
    print(f"  {len(monthly)} monthly rows, {monthly.shape[1]} series  "
          f"({monthly.index[0].date()} → {monthly.index[-1].date()})")
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def mom(s: pd.Series, window: int) -> pd.Series:
    return s.pct_change(window)


def discretize(s: pd.Series, hi: float, lo: float) -> pd.Series:
    """Map continuous signal → +1 / 0 / -1."""
    return pd.Series(
        np.where(s > hi, 1, np.where(s < lo, -1, 0)),
        index=s.index, dtype=float
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. THREE-PILLAR SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame(index=df.index)

    # ── PILLAR 1: MACRO  (Copper / Gold ratio) ───────────────────────────────
    # Copper rises when industry is expanding; Gold rises in fear/deflation.
    # A rising Cu/Au ratio signals broadening economic activity → bullish.
    cu_au = df["copper"] / df["gold"]

    p1a = discretize(mom(cu_au, 6),  hi= 0.04, lo=-0.04)   # 6-month momentum
    p1b = discretize(zscore(cu_au, 24), hi= 0.35, lo=-0.35) # 2-year z-score
    sig["MACRO_RAW"]  = (p1a + p1b) / 2
    sig["MACRO"]      = discretize(sig["MACRO_RAW"], hi=0.25, lo=-0.25)

    # ── PILLAR 2: LIQUIDITY ──────────────────────────────────────────────────
    # Three components, equally weighted:
    #   a) EUR/USD 6-month momentum: rising EUR = risk-on / ECB easing → bullish
    #   b) VIX 2-year z-score (inverted): low VIX = benign conditions → bullish
    #   c) HYG/LQD momentum: HY tightening vs IG = credit risk-on → bullish
    p2a = discretize(mom(df["eurusd"], 6),             hi= 0.025, lo=-0.025)
    p2b = discretize(-zscore(df["vix"], 24),            hi= 0.35,  lo=-0.35)
    p2c = discretize(mom(df["hyg"] / df["lqd"], 3),    hi= 0.01,  lo=-0.01)
    sig["LIQUIDITY_RAW"] = (p2a + p2b + p2c) / 3
    sig["LIQUIDITY"]     = discretize(sig["LIQUIDITY_RAW"], hi=0.2, lo=-0.2)

    # ── PILLAR 3: EPS / EARNINGS ─────────────────────────────────────────────
    # Three components, equally weighted:
    #   a) XLY/XLP (Cyclicals / Defensives) 6-month momentum — earnings cycle
    #   b) XLY/XLP 2-year z-score — structural earnings premium
    #   c) Rate-change drag: sharply rising yields compress equity multiples
    cyc = df["xly"] / df["xlp"]
    p3a = discretize(mom(cyc, 6),          hi= 0.04, lo=-0.04)
    p3b = discretize(zscore(cyc, 24),      hi= 0.35, lo=-0.35)
    rate_chg = (df["tnx"] / 100).diff(6)   # 6-month change in 10Y yield
    p3c = discretize(-rate_chg,             hi= 0.003, lo=-0.003)  # inverted
    sig["EPS_RAW"]  = (p3a + p3b + p3c) / 3
    sig["EPS"]      = discretize(sig["EPS_RAW"], hi=0.2, lo=-0.2)

    # ── COMPOSITE ────────────────────────────────────────────────────────────
    sig["COMPOSITE"] = sig["MACRO"] + sig["LIQUIDITY"] + sig["EPS"]

    # Binary position: 1 (100 % DAX) or 0 (cash), shifted 1 month for realism
    sig["POSITION"] = (sig["COMPOSITE"] >= THRESHOLD).astype(float).shift(1)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index)
    res["dax_ret"]   = df["dax"].pct_change()
    res["position"]  = sig["POSITION"]
    res["composite"] = sig["COMPOSITE"]
    res["strat_ret"] = res["position"] * res["dax_ret"]

    res["dax_cum"]   = (1 + res["dax_ret"]).cumprod()
    res["strat_cum"] = (1 + res["strat_ret"]).cumprod()
    return res


def extract_trades(sig: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    pos = sig["POSITION"].dropna()
    trades, in_trade, entry_date, entry_px = [], False, None, None

    for date, p in pos.items():
        if not in_trade and p == 1:
            in_trade, entry_date, entry_px = True, date, df.loc[date, "dax"]
        elif in_trade and p == 0:
            exit_px = df.loc[date, "dax"]
            trades.append({
                "Entry Date":       entry_date.strftime("%Y-%m-%d"),
                "Exit Date":        date.strftime("%Y-%m-%d"),
                "Entry Price":      round(entry_px, 0),
                "Exit Price":       round(exit_px, 0),
                "Return (%)":       round((exit_px / entry_px - 1) * 100, 2),
                "Duration (mo)":    round((date - entry_date).days / 30.44, 1),
            })
            in_trade = False

    if in_trade:
        exit_px = df["dax"].iloc[-1]
        trades.append({
            "Entry Date":   entry_date.strftime("%Y-%m-%d"),
            "Exit Date":    pos.index[-1].strftime("%Y-%m-%d") + " *open*",
            "Entry Price":  round(entry_px, 0),
            "Exit Price":   round(exit_px, 0),
            "Return (%)":   round((exit_px / entry_px - 1) * 100, 2),
            "Duration (mo)": round((pos.index[-1] - entry_date).days / 30.44, 1),
        })
    return pd.DataFrame(trades)


def performance_stats(res: pd.DataFrame) -> pd.DataFrame:
    def sharpe(r):
        r = r.dropna()
        return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0

    def cagr(cum):
        cum = cum.dropna()
        n_yr = len(cum) / 12
        return (cum.iloc[-1] ** (1 / n_yr) - 1) * 100

    def max_dd(cum):
        cum = cum.dropna()
        return ((cum / cum.cummax()) - 1).min() * 100

    stats = {}
    for label, ret_col, cum_col in [
        ("Strategy",      "strat_ret", "strat_cum"),
        ("DAX Buy&Hold",  "dax_ret",   "dax_cum"),
    ]:
        r, c = res[ret_col], res[cum_col]
        nonzero = r[r != 0].dropna()
        stats[label] = {
            "CAGR (%)":          round(cagr(c),    2),
            "Sharpe Ratio":      round(sharpe(r),  2),
            "Max Drawdown (%)":  round(max_dd(c),  2),
            "Win Rate (%)":      round((nonzero > 0).mean() * 100, 1),
            "Months Invested":   f"{(res['position'] == 1).mean() * 100:.1f}%",
        }
    return pd.DataFrame(stats)


def annual_returns(res: pd.DataFrame) -> pd.DataFrame:
    ann = res[["strat_ret", "dax_ret"]].dropna()
    ann = ann.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    ann.columns = ["Strategy (%)", "DAX B&H (%)"]
    ann.index = ann.index.year
    ann["Alpha (%)"] = (ann["Strategy (%)"] - ann["DAX B&H (%)"]).round(1)
    return ann.round(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(df, sig, res, trades, ann):
    fig = plt.figure(figsize=(18, 24))
    gs  = GridSpec(5, 1, figure=fig, hspace=0.45, top=0.95, bottom=0.04)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])

    GREEN, BLUE, RED = "#27ae60", "#2980b9", "#c0392b"

    # ── 1: Cumulative PnL ────────────────────────────────────────────────────
    ax1.plot(res.index, res["strat_cum"], color=GREEN, lw=2.5, label="MLE Strategy")
    ax1.plot(res.index, res["dax_cum"],  color=BLUE,  lw=2.0, alpha=0.75, label="DAX Buy & Hold")

    # shade invested periods
    pos = res["position"].fillna(0)
    for i in range(len(pos) - 1):
        if pos.iloc[i] == 1:
            ax1.axvspan(pos.index[i], pos.index[i + 1],
                        alpha=0.08, color=GREEN, linewidth=0)

    ax1.set_title("Macro-Liquidity-EPS (MLE) DAX Switching — Cumulative PnL",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (base = 1.0)")
    ax1.legend(fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax1.grid(True, alpha=0.25)

    # ── 2: Drawdowns ─────────────────────────────────────────────────────────
    dd_s = res["strat_cum"] / res["strat_cum"].cummax() - 1
    dd_b = res["dax_cum"]  / res["dax_cum"].cummax()  - 1
    ax2.fill_between(res.index, dd_s * 100, 0, color=RED,    alpha=0.55, label="Strategy DD")
    ax2.fill_between(res.index, dd_b * 100, 0, color="grey", alpha=0.35, label="DAX DD")
    ax2.set_title("Drawdowns", fontsize=13)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    # ── 3: Composite Score & Position ────────────────────────────────────────
    ax3.bar(sig.index, sig["MACRO"],     width=22, alpha=0.55, color="#e67e22", label="Macro pillar")
    ax3.bar(sig.index, sig["LIQUIDITY"], width=22, alpha=0.55, color="#8e44ad", label="Liquidity pillar")
    ax3.bar(sig.index, sig["EPS"],       width=22, alpha=0.55, color="#16a085", label="EPS pillar")
    ax3.plot(sig.index, sig["COMPOSITE"], color="black", lw=2.2, label="Composite")
    ax3.axhline( THRESHOLD, color=GREEN, ls="--", lw=1.8, label=f"Long threshold ({THRESHOLD})")
    ax3.axhline(-THRESHOLD, color=RED,   ls="--", lw=1.2, alpha=0.6)
    ax3.set_title("Pillar Scores & Composite Signal", fontsize=13)
    ax3.set_ylabel("Score (−3 → +3)")
    ax3.set_ylim(-3.8, 3.8)
    ax3.legend(fontsize=9, ncol=3)
    ax3.grid(True, alpha=0.25)

    # ── 4: Raw Indicator Drivers ──────────────────────────────────────────────
    cu_au_mom   = (df["copper"] / df["gold"]).pct_change(6).reindex(sig.index)
    vix_z_neg   = (-zscore(df["vix"], 24) * 0.06).reindex(sig.index)
    cyc_mom     = (df["xly"] / df["xlp"]).pct_change(6).reindex(sig.index)
    eurusd_mom  = df["eurusd"].pct_change(6).reindex(sig.index)

    ax4.plot(sig.index, cu_au_mom,  color="#e67e22", lw=1.6, label="Cu/Au 6M mom (Macro)")
    ax4.plot(sig.index, vix_z_neg,  color="#8e44ad", lw=1.6, label="−VIX z-score ×0.06 (Liq.)")
    ax4.plot(sig.index, cyc_mom,    color="#16a085", lw=1.6, label="XLY/XLP 6M mom (EPS)")
    ax4.plot(sig.index, eurusd_mom, color="#2980b9", lw=1.3, alpha=0.8, label="EUR/USD 6M mom (Liq.)")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_title("Raw Indicator Drivers", fontsize=13)
    ax4.set_ylabel("Value")
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.25)

    # ── 5: Annual Return Bar Chart ────────────────────────────────────────────
    x     = np.arange(len(ann))
    width = 0.38
    bars_s = ax5.bar(x - width / 2, ann["Strategy (%)"],  width, color=GREEN, alpha=0.80, label="Strategy")
    bars_b = ax5.bar(x + width / 2, ann["DAX B&H (%)"],   width, color=BLUE,  alpha=0.65, label="DAX B&H")
    ax5.set_xticks(x)
    ax5.set_xticklabels(ann.index.astype(str), rotation=45, fontsize=9)
    ax5.axhline(0, color="black", lw=0.8)
    ax5.set_title("Annual Returns — Strategy vs DAX Buy & Hold", fontsize=13)
    ax5.set_ylabel("Return (%)")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.25, axis="y")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.savefig("mle_strategy_backtest.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("\n  Chart saved → mle_strategy_backtest.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP = "─" * 68

    print()
    print("=" * 68)
    print("   MACRO-LIQUIDITY-EPS (MLE) DAX SWITCHING STRATEGY — BACKTEST")
    print("=" * 68)
    print(f"  Period: {START} → {END}   |   Rebalancing: monthly   |   Threshold: {THRESHOLD}")
    print()

    df  = download_data()
    sig = build_signals(df)
    res = run_backtest(df, sig)
    trades = extract_trades(sig, df)
    stats  = performance_stats(res)
    ann    = annual_returns(res)

    # ── Performance Summary ────────────────────────────────────────────────
    print(SEP)
    print("  PERFORMANCE SUMMARY")
    print(SEP)
    print(stats.to_string())

    # ── Annual Returns ─────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  ANNUAL RETURNS")
    print(SEP)
    print(ann.to_string())

    # ── Trade Log ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  TRADE LOG  ({len(trades)} trades)")
    print(SEP)
    if trades.empty:
        print("  No completed trades found.")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.2f}".format)
        print(trades.to_string(index=False))

        wins  = trades[~trades["Exit Date"].str.contains("open")]["Return (%)"]
        print()
        print(f"  Completed trades : {len(wins)}")
        print(f"  Win rate         : {(wins > 0).mean() * 100:.1f}%")
        print(f"  Avg return/trade : {wins.mean():.2f}%")
        print(f"  Best trade       : {wins.max():.2f}%")
        print(f"  Worst trade      : {wins.min():.2f}%")

    # ── Charts ─────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  Generating charts …")
    plot_all(df, sig, res, trades, ann)

    print()
    print("=" * 68)
    print("  Done.")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
