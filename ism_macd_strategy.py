"""
ISM Manufacturing MACD — DAX Switching Strategy
================================================

Signal source : ISM Manufacturing PMI (monthly, embedded from public data)
Instrument    : DAX (^GDAXI, total-return not adjusted for dividends)
Rebalancing   : Monthly

─── SIGNAL LOGIC ────────────────────────────────────────────────────────────
BUY  : "Early indication of MACD bottoming"
       → After BUY_LOOKBACK consecutive declining MACD-histogram bars,
         the histogram makes its FIRST uptick (bar[i] > bar[i-1]).
         This catches the earliest possible turning point.

SELL : "5-6 MACD candles falling from top"
       → SELL_BARS consecutive declining MACD-histogram bars
         (each bar strictly lower than the previous).
         This detects sustained deterioration from a local peak.

Positions are executed with a 1-month lag to avoid look-ahead bias.

─── OPTIMIZATION ────────────────────────────────────────────────────────────
Grid search over:
  fast_period  ∈ {3, 5, 6, 9, 12}
  slow_period  ∈ {9, 12, 18, 21, 26}
  signal_period∈ {3, 4, 5, 6, 9}
  buy_lookback ∈ {1, 2, 3, 4}        (bars declining before uptick triggers buy)
  sell_bars    ∈ {3, 4, 5, 6, 7, 8}  (consecutive declining bars to sell)

Objective: maximise Sharpe Ratio (annualised, monthly data)
Secondary:  maximise Calmar Ratio (CAGR / |Max Drawdown|)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EMBEDDED ISM MANUFACTURING PMI DATA  (NAPM / ISM, monthly, 2000 – 2024)
#     Source: ISM Institute for Supply Management (public monthly releases)
# ─────────────────────────────────────────────────────────────────────────────

ISM_CSV = """\
date,ism
2000-01-31,56.3
2000-02-29,56.7
2000-03-31,55.8
2000-04-30,54.9
2000-05-31,53.2
2000-06-30,51.8
2000-07-31,51.8
2000-08-31,49.9
2000-09-30,49.9
2000-10-31,48.4
2000-11-30,48.2
2000-12-31,43.2
2001-01-31,41.2
2001-02-28,41.9
2001-03-31,47.5
2001-04-30,43.9
2001-05-31,42.1
2001-06-30,44.7
2001-07-31,43.5
2001-08-31,47.9
2001-09-30,47.1
2001-10-31,39.8
2001-11-30,44.5
2001-12-31,48.2
2002-01-31,49.9
2002-02-28,54.7
2002-03-31,55.6
2002-04-30,53.9
2002-05-31,56.0
2002-06-30,56.2
2002-07-31,50.5
2002-08-31,50.5
2002-09-30,49.5
2002-10-31,48.7
2002-11-30,49.2
2002-12-31,54.7
2003-01-31,53.9
2003-02-28,50.5
2003-03-31,46.2
2003-04-30,45.4
2003-05-31,49.4
2003-06-30,52.7
2003-07-31,51.8
2003-08-31,55.3
2003-09-30,53.6
2003-10-31,57.0
2003-11-30,62.8
2003-12-31,66.2
2004-01-31,63.6
2004-02-29,61.4
2004-03-31,62.5
2004-04-30,62.4
2004-05-31,62.8
2004-06-30,61.1
2004-07-31,60.5
2004-08-31,59.0
2004-09-30,58.5
2004-10-31,56.8
2004-11-30,57.8
2004-12-31,57.6
2005-01-31,57.3
2005-02-28,55.3
2005-03-31,55.2
2005-04-30,53.3
2005-05-31,51.4
2005-06-30,53.8
2005-07-31,56.6
2005-08-31,53.6
2005-09-30,59.4
2005-10-31,59.1
2005-11-30,58.1
2005-12-31,55.6
2006-01-31,56.8
2006-02-28,56.7
2006-03-31,55.9
2006-04-30,57.3
2006-05-31,54.4
2006-06-30,53.8
2006-07-31,54.7
2006-08-31,54.5
2006-09-30,52.9
2006-10-31,51.2
2006-11-30,49.5
2006-12-31,51.4
2007-01-31,49.3
2007-02-28,52.3
2007-03-31,50.9
2007-04-30,54.7
2007-05-31,52.9
2007-06-30,56.0
2007-07-31,53.8
2007-08-31,52.9
2007-09-30,52.0
2007-10-31,50.9
2007-11-30,50.8
2007-12-31,48.4
2008-01-31,50.7
2008-02-29,48.3
2008-03-31,48.6
2008-04-30,48.6
2008-05-31,49.6
2008-06-30,50.2
2008-07-31,50.0
2008-08-31,49.9
2008-09-30,43.5
2008-10-31,38.9
2008-11-30,36.2
2008-12-31,32.4
2009-01-31,35.6
2009-02-28,35.8
2009-03-31,36.3
2009-04-30,40.1
2009-05-31,42.8
2009-06-30,44.8
2009-07-31,48.9
2009-08-31,52.9
2009-09-30,52.6
2009-10-31,55.7
2009-11-30,53.6
2009-12-31,55.9
2010-01-31,58.4
2010-02-28,56.5
2010-03-31,59.6
2010-04-30,60.4
2010-05-31,59.7
2010-06-30,56.2
2010-07-31,55.5
2010-08-31,56.3
2010-09-30,54.4
2010-10-31,56.9
2010-11-30,56.6
2010-12-31,57.0
2011-01-31,60.8
2011-02-28,61.4
2011-03-31,61.2
2011-04-30,60.4
2011-05-31,53.5
2011-06-30,50.9
2011-07-31,50.9
2011-08-31,50.6
2011-09-30,51.6
2011-10-31,50.8
2011-11-30,52.7
2011-12-31,53.9
2012-01-31,54.1
2012-02-29,52.4
2012-03-31,53.4
2012-04-30,54.8
2012-05-31,53.5
2012-06-30,49.7
2012-07-31,49.8
2012-08-31,49.6
2012-09-30,51.5
2012-10-31,51.7
2012-11-30,49.5
2012-12-31,50.7
2013-01-31,53.1
2013-02-28,54.2
2013-03-31,51.3
2013-04-30,50.7
2013-05-31,49.0
2013-06-30,50.9
2013-07-31,55.4
2013-08-31,55.7
2013-09-30,56.2
2013-10-31,56.4
2013-11-30,57.3
2013-12-31,57.0
2014-01-31,51.3
2014-02-28,53.2
2014-03-31,53.7
2014-04-30,54.9
2014-05-31,55.4
2014-06-30,55.3
2014-07-31,57.1
2014-08-31,59.0
2014-09-30,56.6
2014-10-31,59.0
2014-11-30,58.7
2014-12-31,55.5
2015-01-31,53.5
2015-02-28,52.9
2015-03-31,51.5
2015-04-30,51.5
2015-05-31,52.8
2015-06-30,53.5
2015-07-31,52.7
2015-08-31,51.1
2015-09-30,50.2
2015-10-31,50.1
2015-11-30,48.6
2015-12-31,48.2
2016-01-31,48.2
2016-02-29,49.5
2016-03-31,51.8
2016-04-30,50.8
2016-05-31,51.3
2016-06-30,53.2
2016-07-31,52.6
2016-08-31,49.4
2016-09-30,51.5
2016-10-31,51.9
2016-11-30,53.2
2016-12-31,54.7
2017-01-31,56.0
2017-02-28,57.7
2017-03-31,57.2
2017-04-30,54.8
2017-05-31,54.9
2017-06-30,57.8
2017-07-31,56.3
2017-08-31,58.8
2017-09-30,60.8
2017-10-31,58.7
2017-11-30,58.2
2017-12-31,59.7
2018-01-31,59.1
2018-02-28,60.8
2018-03-31,59.3
2018-04-30,57.3
2018-05-31,58.7
2018-06-30,60.2
2018-07-31,58.1
2018-08-31,61.3
2018-09-30,59.8
2018-10-31,57.7
2018-11-30,59.3
2018-12-31,54.1
2019-01-31,56.6
2019-02-28,54.2
2019-03-31,55.3
2019-04-30,52.8
2019-05-31,52.1
2019-06-30,51.7
2019-07-31,51.2
2019-08-31,49.1
2019-09-30,47.8
2019-10-31,48.3
2019-11-30,48.1
2019-12-31,47.2
2020-01-31,50.9
2020-02-29,50.1
2020-03-31,49.1
2020-04-30,41.5
2020-05-31,43.1
2020-06-30,52.6
2020-07-31,54.2
2020-08-31,56.0
2020-09-30,55.4
2020-10-31,59.3
2020-11-30,57.5
2020-12-31,60.7
2021-01-31,58.7
2021-02-28,60.8
2021-03-31,64.7
2021-04-30,60.7
2021-05-31,61.2
2021-06-30,60.6
2021-07-31,59.5
2021-08-31,59.9
2021-09-30,61.1
2021-10-31,60.8
2021-11-30,61.1
2021-12-31,58.7
2022-01-31,57.6
2022-02-28,58.6
2022-03-31,57.1
2022-04-30,55.4
2022-05-31,56.1
2022-06-30,53.0
2022-07-31,52.8
2022-08-31,52.8
2022-09-30,50.9
2022-10-31,50.2
2022-11-30,49.0
2022-12-31,48.4
2023-01-31,47.4
2023-02-28,47.7
2023-03-31,46.3
2023-04-30,47.1
2023-05-31,46.9
2023-06-30,46.0
2023-07-31,46.4
2023-08-31,47.6
2023-09-30,49.0
2023-10-31,46.7
2023-11-30,46.7
2023-12-31,47.4
2024-01-31,49.1
2024-02-29,47.8
2024-03-31,50.3
2024-04-30,49.2
2024-05-31,48.7
2024-06-30,48.5
2024-07-31,46.8
2024-08-31,47.2
2024-09-30,47.2
2024-10-31,46.5
2024-11-30,48.4
2024-12-31,49.3
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
START = "2000-01-01"
END   = "2024-12-31"
REBAL = "ME"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_ism() -> pd.Series:
    df = pd.read_csv(io.StringIO(ISM_CSV), parse_dates=["date"], index_col="date")
    df.index = df.index.to_period("M").to_timestamp("M")   # month-end index
    return df["ism"].sort_index()


def load_dax() -> pd.Series:
    raw = yf.download("^GDAXI", start=START, end=END,
                      auto_adjust=True, progress=False)["Close"]
    # yfinance may return DataFrame for single ticker — squeeze to Series
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()
    return raw.resample(REBAL).last().ffill(limit=5)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MACD
# ─────────────────────────────────────────────────────────────────────────────
def macd(series: pd.Series, fast: int, slow: int, sig: int):
    """Returns histogram = (EMA_fast - EMA_slow) - EMA_signal."""
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_raw_position(hist: np.ndarray,
                           buy_lookback: int,
                           sell_bars: int) -> np.ndarray:
    """
    State-machine on the MACD histogram array.

    BUY  : first uptick (h[i] > h[i-1]) after buy_lookback consecutive
           declining bars  (h[i-1] > h[i-2] > … > h[i-buy_lookback-1])
    SELL : sell_bars consecutive declining bars
           (h[i] < h[i-1] < … < h[i-sell_bars+1] < h[i-sell_bars])
    """
    n = len(hist)
    pos = np.zeros(n)
    in_trade = False
    start = max(sell_bars, buy_lookback) + 2

    for i in range(start, n):
        h = hist
        if not in_trade:
            # Need uptick at i AND buy_lookback declining bars before i
            uptick = h[i] > h[i - 1]
            if uptick and buy_lookback > 0:
                prev_decline = all(h[i - 1 - k] < h[i - 2 - k]
                                   for k in range(buy_lookback))
                if prev_decline:
                    in_trade = True
                    pos[i] = 1
            elif uptick and buy_lookback == 0:
                in_trade = True
                pos[i] = 1
        else:
            pos[i] = 1
            # Check sell: sell_bars consecutive declines ending at i
            falling = all(h[i - k] < h[i - k - 1]
                          for k in range(sell_bars))
            if falling:
                in_trade = False
                pos[i] = 0   # exit this bar

    return pos


def make_position(ism: pd.Series, fast: int, slow: int, sig: int,
                  buy_lookback: int, sell_bars: int) -> pd.Series:
    _, _, hist = macd(ism, fast, slow, sig)
    raw = generate_raw_position(hist.values, buy_lookback, sell_bars)
    pos = pd.Series(raw, index=ism.index, dtype=float)
    # 1-month lag → position is set at month-end, executed the following month
    return pos.shift(1).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    return (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0


def cagr(cum: pd.Series) -> float:
    c = cum.dropna()
    if len(c) < 2:
        return 0.0
    n_yr = len(c) / 12
    return (c.iloc[-1] ** (1 / n_yr) - 1) * 100


def max_dd(cum: pd.Series) -> float:
    c = cum.dropna()
    return ((c / c.cummax()) - 1).min() * 100


def calmar(cum: pd.Series) -> float:
    c = cagr(cum)
    d = abs(max_dd(cum))
    return c / d if d > 0 else 0.0


def win_rate(rets: pd.Series) -> float:
    r = rets[rets != 0].dropna()
    return (r > 0).mean() * 100 if len(r) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRID SEARCH OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
FAST_VALS   = [3, 5, 6, 9, 12]
SLOW_VALS   = [9, 12, 18, 21, 26]
SIG_VALS    = [3, 4, 5, 6, 9]
BUY_LB_VALS = [1, 2, 3, 4]
SELL_VALS   = [3, 4, 5, 6, 7, 8]


def optimise(ism: pd.Series, dax_ret: pd.Series) -> pd.DataFrame:
    rows = []
    combos = [(f, s, sg, b, sv)
              for f in FAST_VALS
              for s in SLOW_VALS
              for sg in SIG_VALS
              for b in BUY_LB_VALS
              for sv in SELL_VALS
              if f < s]                    # constraint: fast < slow

    print(f"  Grid search: {len(combos)} parameter combinations …")

    for fast, slow, sig, buy_lb, sell_b in combos:
        pos = make_position(ism, fast, slow, sig, buy_lb, sell_b)
        # Align on common index
        common = pos.index.intersection(dax_ret.index)
        pos_c  = pos.reindex(common)
        ret_c  = dax_ret.reindex(common)

        strat_ret = pos_c * ret_c
        cum       = (1 + strat_ret).cumprod()
        n_months  = (pos_c == 1).sum()

        sh = sharpe(strat_ret)
        ca = cagr(cum)
        md = max_dd(cum)
        cl = calmar(cum)

        rows.append({
            "fast": fast, "slow": slow, "signal": sig,
            "buy_lookback": buy_lb, "sell_bars": sell_b,
            "sharpe": round(sh, 3),
            "cagr_%": round(ca, 2),
            "max_dd_%": round(md, 2),
            "calmar": round(cl, 3),
            "win_rate_%": round(win_rate(strat_ret), 1),
            "months_invested": int(n_months),
        })

    results = pd.DataFrame(rows)
    return results.sort_values("sharpe", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL BACKTEST WITH TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(ism: pd.Series, dax: pd.Series,
                 fast: int, slow: int, sig: int,
                 buy_lb: int, sell_b: int):

    pos = make_position(ism, fast, slow, sig, buy_lb, sell_b)
    common = pos.index.intersection(dax.dropna().index)
    pos    = pos.reindex(common)
    dax_c  = dax.reindex(common)

    res = pd.DataFrame(index=common)
    res["dax_ret"]   = dax_c.pct_change()
    res["position"]  = pos
    res["strat_ret"] = pos * res["dax_ret"]
    res["dax_cum"]   = (1 + res["dax_ret"]).cumprod()
    res["strat_cum"] = (1 + res["strat_ret"]).cumprod()

    _, _, hist_s = macd(ism, fast, slow, sig)
    res["macd_hist"] = hist_s.reindex(common)

    return res


def extract_trades(res: pd.DataFrame, dax: pd.Series) -> pd.DataFrame:
    pos = res["position"]
    trades, in_trade = [], False
    entry_date = entry_px = None

    for date, p in pos.items():
        if not in_trade and p == 1:
            in_trade, entry_date = True, date
            entry_px = dax.get(date, np.nan)
        elif in_trade and p == 0:
            exit_px = dax.get(date, np.nan)
            pct = (exit_px / entry_px - 1) * 100 if entry_px and not np.isnan(entry_px) else np.nan
            trades.append({
                "Entry Date":      entry_date.strftime("%Y-%m-%d"),
                "Exit Date":       date.strftime("%Y-%m-%d"),
                "Entry DAX":       round(entry_px, 0),
                "Exit DAX":        round(exit_px, 0),
                "Return (%)":      round(pct, 2),
                "Duration (mo)":   round((date - entry_date).days / 30.44, 1),
            })
            in_trade = False

    if in_trade:
        last_date = pos.index[-1]
        exit_px   = dax.get(last_date, np.nan)
        pct = (exit_px / entry_px - 1) * 100 if entry_px and not np.isnan(entry_px) else np.nan
        trades.append({
            "Entry Date":  entry_date.strftime("%Y-%m-%d"),
            "Exit Date":   last_date.strftime("%Y-%m-%d") + " *open*",
            "Entry DAX":   round(entry_px, 0),
            "Exit DAX":    round(exit_px, 0),
            "Return (%)":  round(pct, 2),
            "Duration (mo)": round((last_date - entry_date).days / 30.44, 1),
        })
    return pd.DataFrame(trades)


def annual_returns(res: pd.DataFrame) -> pd.DataFrame:
    ann = (res[["strat_ret", "dax_ret"]].dropna()
           .resample("YE")
           .apply(lambda x: (1 + x).prod() - 1) * 100)
    ann.columns = ["Strategy (%)", "DAX B&H (%)"]
    ann.index = ann.index.year
    ann["Alpha (%)"] = (ann["Strategy (%)"] - ann["DAX B&H (%)"]).round(1)
    return ann.round(1)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(ism, res, trades, ann, params):
    fast, slow, sig, buy_lb, sell_b = params
    fig = plt.figure(figsize=(18, 26))
    gs  = GridSpec(5, 1, figure=fig, hspace=0.42, top=0.95, bottom=0.04)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])

    GREEN, BLUE, RED, ORG = "#27ae60", "#2980b9", "#c0392b", "#e67e22"

    # ── Chart 1 : Cumulative PnL ───────────────────────────────────────────
    ax1.plot(res.index, res["strat_cum"], color=GREEN, lw=2.5, label="ISM-MACD Strategy")
    ax1.plot(res.index, res["dax_cum"],  color=BLUE,  lw=2.0, alpha=0.7, label="DAX Buy & Hold")

    pos = res["position"].fillna(0)
    for i in range(len(pos) - 1):
        if pos.iloc[i] == 1:
            ax1.axvspan(pos.index[i], pos.index[i + 1],
                        alpha=0.08, color=GREEN, linewidth=0)

    ax1.set_title(
        f"ISM Manufacturing MACD — DAX Switching Strategy\n"
        f"MACD({fast},{slow},{sig})  |  Buy after {buy_lb} declining bars + uptick  |  Sell after {sell_b} consecutive declines",
        fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Portfolio Value (base = 1.0)")
    ax1.legend(fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax1.grid(True, alpha=0.25)

    # ── Chart 2 : Drawdowns ────────────────────────────────────────────────
    dd_s = res["strat_cum"] / res["strat_cum"].cummax() - 1
    dd_b = res["dax_cum"]  / res["dax_cum"].cummax()  - 1
    ax2.fill_between(res.index, dd_s * 100, 0, color=RED,    alpha=0.6, label="Strategy DD")
    ax2.fill_between(res.index, dd_b * 100, 0, color="grey", alpha=0.35, label="DAX DD")
    ax2.set_title("Drawdowns", fontsize=12)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    # ── Chart 3 : ISM PMI + MACD ──────────────────────────────────────────
    ax3_r = ax3.twinx()
    ism_plot = ism.reindex(res.index)
    hist_plot = res["macd_hist"]

    ax3.plot(res.index, ism_plot, color=ORG, lw=1.8, label="ISM PMI", alpha=0.9)
    ax3.axhline(50, color=ORG, ls="--", lw=1.2, alpha=0.6)
    ax3.set_ylabel("ISM PMI", color=ORG)
    ax3.tick_params(axis="y", colors=ORG)
    ax3.set_ylim(28, 72)

    colors = [GREEN if v >= 0 else RED for v in hist_plot.fillna(0)]
    ax3_r.bar(res.index, hist_plot, color=colors, alpha=0.6, width=22,
              label="MACD Histogram")
    ax3_r.set_ylabel("MACD Histogram", color="grey")
    ax3_r.axhline(0, color="black", lw=0.8)

    # Mark buy/sell points on histogram
    buy_dates  = [trades.loc[i, "Entry Date"] for i in trades.index
                  if "*open*" not in str(trades.loc[i, "Exit Date"])]
    sell_dates = [trades.loc[i, "Exit Date"]  for i in trades.index
                  if "*open*" not in str(trades.loc[i, "Exit Date"])]
    for d in buy_dates:
        try:
            dt = pd.Timestamp(d)
            if dt in hist_plot.index:
                ax3_r.scatter(dt, hist_plot.loc[dt], marker="^",
                              color=GREEN, s=80, zorder=5)
        except Exception:
            pass
    for d in sell_dates:
        try:
            dt = pd.Timestamp(d)
            if dt in hist_plot.index:
                ax3_r.scatter(dt, hist_plot.loc[dt], marker="v",
                              color=RED, s=80, zorder=5)
        except Exception:
            pass

    ax3.set_title(f"ISM PMI & MACD Histogram  (▲ = buy signal, ▼ = sell signal)", fontsize=12)
    lines1, lbl1 = ax3.get_legend_handles_labels()
    lines2, lbl2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.2)

    # ── Chart 4 : Annual Returns bar ──────────────────────────────────────
    x     = np.arange(len(ann))
    width = 0.38
    ax4.bar(x - width / 2, ann["Strategy (%)"], width, color=GREEN, alpha=0.8, label="Strategy")
    ax4.bar(x + width / 2, ann["DAX B&H (%)"],  width, color=BLUE,  alpha=0.65, label="DAX B&H")
    ax4.set_xticks(x)
    ax4.set_xticklabels(ann.index.astype(str), rotation=45, fontsize=9)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_title("Annual Returns — Strategy vs DAX Buy & Hold", fontsize=12)
    ax4.set_ylabel("Return (%)")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.25, axis="y")

    # ── Chart 5 : Trade returns bar chart ────────────────────────────────
    completed = trades[~trades["Exit Date"].str.contains("open")].copy()
    if not completed.empty:
        t_colors = [GREEN if r > 0 else RED for r in completed["Return (%)"].values]
        ax5.bar(range(len(completed)), completed["Return (%)"].values,
                color=t_colors, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax5.axhline(0, color="black", lw=0.8)
        ax5.set_xticks(range(len(completed)))
        ax5.set_xticklabels(
            [f"{r['Entry Date'][:7]}" for _, r in completed.iterrows()],
            rotation=45, fontsize=8
        )
        ax5.set_title(
            f"Individual Trade Returns  ({len(completed)} completed trades | "
            f"Win rate: {(completed['Return (%)'] > 0).mean() * 100:.1f}%)",
            fontsize=12
        )
        ax5.set_ylabel("Return (%)")
        ax5.grid(True, alpha=0.25, axis="y")

    for ax in [ax1, ax2, ax3, ax4]:
        if ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.savefig("ism_macd_backtest.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Chart saved → ism_macd_backtest.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP  = "─" * 72
    SEP2 = "═" * 72

    print()
    print(SEP2)
    print("   ISM MANUFACTURING MACD — DAX SWITCHING STRATEGY")
    print(SEP2)
    print(f"   Period: {START} → {END}   |   Rebalancing: monthly   |   Lag: 1 month")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    print("  Loading ISM PMI data …")
    ism = load_ism()
    print(f"  ISM: {len(ism)} months  ({ism.index[0].date()} → {ism.index[-1].date()})"
          f"  range [{ism.min():.1f}, {ism.max():.1f}]")

    print("  Downloading DAX (^GDAXI) via yfinance …")
    dax = load_dax()
    dax_ret = dax.pct_change()
    print(f"  DAX: {len(dax)} months  ({dax.index[0].date()} → {dax.index[-1].date()})")

    # ── B&H benchmark ─────────────────────────────────────────────────────
    bnh_cum  = (1 + dax_ret).cumprod().dropna()
    bnh_cagr = cagr(bnh_cum)
    bnh_sh   = sharpe(dax_ret)
    bnh_dd   = max_dd(bnh_cum)

    print()
    print(SEP)
    print(f"  BENCHMARK  DAX Buy & Hold: "
          f"CAGR {bnh_cagr:.1f}%  |  Sharpe {bnh_sh:.2f}  |  Max DD {bnh_dd:.1f}%")
    print(SEP)

    # ── Grid search ────────────────────────────────────────────────────────
    print()
    opt = optimise(ism, dax_ret)

    print()
    print(SEP)
    print("  TOP 15 PARAMETER COMBINATIONS  (sorted by Sharpe)")
    print(SEP)
    print(opt.head(15).to_string(index=True))

    # ── Best parameters ────────────────────────────────────────────────────
    best = opt.iloc[0]
    best_params = (int(best["fast"]), int(best["slow"]), int(best["signal"]),
                   int(best["buy_lookback"]), int(best["sell_bars"]))
    fast, slow, sig, buy_lb, sell_b = best_params

    print()
    print(SEP)
    print(f"  BEST PARAMETERS  →  MACD({fast},{slow},{sig})"
          f"  buy_lookback={buy_lb}  sell_bars={sell_b}")
    print(SEP)

    # ── Full backtest with best params ────────────────────────────────────
    res    = run_backtest(ism, dax, fast, slow, sig, buy_lb, sell_b)
    trades = extract_trades(res, dax)
    ann    = annual_returns(res)

    # ── Performance summary ───────────────────────────────────────────────
    st_cum = res["strat_cum"].dropna()
    bh_cum = res["dax_cum"].dropna()

    summary = pd.DataFrame({
        "Strategy": {
            "CAGR (%)":         round(cagr(st_cum),   2),
            "Sharpe Ratio":     round(sharpe(res["strat_ret"]), 2),
            "Calmar Ratio":     round(calmar(st_cum),  2),
            "Max Drawdown (%)": round(max_dd(st_cum),  2),
            "Win Rate (%)":     round(win_rate(res["strat_ret"]), 1),
            "Months Invested":  f"{(res['position'] == 1).mean() * 100:.1f}%",
        },
        "DAX Buy & Hold": {
            "CAGR (%)":         round(cagr(bh_cum),   2),
            "Sharpe Ratio":     round(sharpe(res["dax_ret"]), 2),
            "Calmar Ratio":     round(calmar(bh_cum),  2),
            "Max Drawdown (%)": round(max_dd(bh_cum),  2),
            "Win Rate (%)":     round(win_rate(res["dax_ret"]), 1),
            "Months Invested":  "100.0%",
        },
    })
    print()
    print(SEP)
    print("  PERFORMANCE SUMMARY")
    print(SEP)
    print(summary.to_string())

    # ── Annual returns ─────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  ANNUAL RETURNS")
    print(SEP)
    print(ann.to_string())

    # ── Trade log ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  TRADE LOG  ({len(trades)} trades total)")
    print(SEP)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(trades.to_string(index=False))

    completed = trades[~trades["Exit Date"].str.contains("open")]
    if not completed.empty:
        wr  = (completed["Return (%)"] > 0).mean() * 100
        avg = completed["Return (%)"].mean()
        best_t  = completed.loc[completed["Return (%)"].idxmax()]
        worst_t = completed.loc[completed["Return (%)"].idxmin()]
        print()
        print(f"  Completed trades : {len(completed)}")
        print(f"  Win rate         : {wr:.1f}%")
        print(f"  Avg return/trade : {avg:.2f}%")
        print(f"  Best  trade      : {best_t['Return (%)']:.2f}%  "
              f"({best_t['Entry Date']} → {best_t['Exit Date']})")
        print(f"  Worst trade      : {worst_t['Return (%)']:.2f}%  "
              f"({worst_t['Entry Date']} → {worst_t['Exit Date']})")

    # ── Charts ─────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  Generating charts …")
    plot_results(ism, res, trades, ann, best_params)

    print()
    print(SEP2)
    print("  Done.")
    print(SEP2)
    print()


if __name__ == "__main__":
    main()
