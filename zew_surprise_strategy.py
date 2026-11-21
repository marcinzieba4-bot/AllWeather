"""
ZEW "Positive Surprise" DAX Strategy
====================================

Idea
----
ZEW Economic Sentiment (Germany, monthly) is at an UPPER level, ticks DOWN
(looks like the start of a negative cycle) and then turns UP again on the next
release.  That re-acceleration is treated as a positive economic surprise.

  SIGNAL  at release k :  ZEW[k] > ZEW[k-1]  and  ZEW[k-1] < ZEW[k-2]
                          and  ZEW[k-1] >= LEVEL        (trough still high)
  FILTER  (price)      :  last completed DAX candle (month / week) closed UP.
                          The signal stays armed until the next ZEW release;
                          the filter is re-checked at every candle completion.
  EXIT                 :  ZEW falls again (1 or 2 releases), or time stop.

Mechanics: daily DAX closes, real ZEW release dates (no look-ahead), every
decision taken at a close is executed at the NEXT close, 0.10% cost per switch,
cash earns 0%.

Data: data/zew_konjunktur.xlsx  (ZEW, ftp.zew.de/pub/zew-docs/div/konjunktur.xlsx)
      data/dax_daily.csv        (^GDAXI performance index, Yahoo Finance)
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZEW_FILE = os.path.join(HERE, "data", "zew_konjunktur.xlsx")
DAX_FILE = os.path.join(HERE, "data", "dax_daily.csv")
ZEW_URL = "https://ftp.zew.de/pub/zew-docs/div/konjunktur.xlsx"

START = "1992-01-01"
SPLIT = "2009-01-01"          # first half / second half robustness split
COST = 0.001                  # per switch
# Chosen specification (thesis-consistent, not the grid maximum)
BASE = dict(L=30, filt="month", arm=1, exit="zew1")


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_zew() -> pd.Series:
    src = ZEW_FILE if os.path.exists(ZEW_FILE) else ZEW_URL
    z = pd.read_excel(src)
    z.columns = ["date", "zew", "situation"]
    z = z[pd.to_datetime(z["date"], errors="coerce").notna()].copy()
    z["date"] = pd.to_datetime(z["date"])          # actual release dates
    return z.set_index("date")["zew"].astype(float)


def load_dax() -> pd.Series:
    if os.path.exists(DAX_FILE):
        px = pd.read_csv(DAX_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        px = yf.download("^GDAXI", start="1988-01-01", progress=False,
                         auto_adjust=True)["Close"].squeeze()
    return px.dropna()[lambda s: s.index >= "1991-06-01"].rename("dax")


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, zew: pd.Series, px: pd.Series):
        self.zew, self.px = zew, px
        self.idx, self.n = px.index, len(px)
        self.ret = px.pct_change().fillna(0).values
        self.zv = zew.values
        self.dz = np.diff(self.zv, prepend=np.nan)
        self.rel_i = self.idx.searchsorted(zew.index)   # first trading day >= release
        self.rel_at = {i: k for k, i in enumerate(self.rel_i) if i < self.n}

        # "last completed weekly / monthly candle closed up", as of each close
        def last_candle_up(freq_period):
            per = self.idx.to_period(freq_period)
            close = px.groupby(per).last()
            up = (close.diff() > 0)
            is_end = np.append(per[1:] != per[:-1], True)
            out, cur = np.zeros(self.n, bool), False
            for i in range(self.n):
                if is_end[i]:
                    cur = bool(up.loc[per[i]])
                out[i] = cur
            return out, is_end
        self.wk_up, self.wk_end = last_candle_up("W-FRI")
        self.mo_up, self.mo_end = last_candle_up("M")

    def signal(self, k: int, L: float) -> bool:
        return (k >= 2 and self.dz[k] > 0 and self.dz[k - 1] < 0
                and self.zv[k - 1] >= L)

    def run(self, L=30, filt="month", arm=1, exit="zew1", hold=6):
        """Returns daily position array (1 = long over the following day)."""
        n, pos = self.n, np.zeros(self.n)
        inpos, pend, armed_until, last_sig, trades, entry = False, None, -1, 0, [], None
        for i in range(1, n):
            if pend == "in" and not inpos:
                inpos, entry = True, i
            elif pend == "out" and inpos:
                inpos = False
                trades.append((entry, i))
            pend = None
            pos[i] = float(inpos)

            k = self.rel_at.get(i)
            if k is not None and k >= 2:
                if self.signal(k, L):
                    last_sig = i
                    nxt = self.rel_i[k + arm] if k + arm < len(self.rel_i) else n
                    armed_until = nxt - 1
                if inpos:
                    if exit == "zew1" and self.dz[k] < 0:
                        pend = "out"
                    elif exit == "zew2" and self.dz[k] < 0 and self.dz[k - 1] < 0:
                        pend = "out"
            if not inpos and pend is None and i <= armed_until:
                if filt == "none":
                    ok = True
                elif filt == "week":
                    ok = self.wk_up[i] and (k is not None or self.wk_end[i])
                else:  # month
                    ok = self.mo_up[i] and (k is not None or self.mo_end[i])
                if ok:
                    pend, armed_until = "in", -1
            if inpos and pend is None and exit == "hold" and i - last_sig >= hold * 21:
                pend = "out"
        if inpos:
            trades.append((entry, n - 1))
        return pos, trades

    def returns(self, pos: np.ndarray) -> pd.Series:
        held = np.roll(pos, 1); held[0] = 0
        turn = np.abs(np.diff(pos, prepend=0))
        return pd.Series(held * self.ret - turn * COST, index=self.idx)


def stats(r: pd.Series) -> dict:
    c = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"CAGR %": (c.iloc[-1] ** (1 / yrs) - 1) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            "MaxDD %": (c / c.cummax() - 1).min() * 100,
            "Invested %": (r != 0).mean() * 100}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSES
# ─────────────────────────────────────────────────────────────────────────────
def event_study(e: Engine) -> pd.DataFrame:
    z, d1 = e.zew, e.zew.diff()
    prev = z.shift(1)
    up, pdn = d1 > 0, d1.shift(1) < 0
    sets = {
        "All releases": pd.Series(True, index=z.index),
        "Rebound, any level": up & pdn,
        "Rebound, trough < 0": up & pdn & (prev < 0),
        "Rebound, trough >= 0": up & pdn & (prev >= 0),
        "Rebound, trough >= 20": up & pdn & (prev >= 20),
        "Rebound, trough >= 30  <- signal": up & pdn & (prev >= 30),
        "Rebound, trough >= 40": up & pdn & (prev >= 40),
        "ZEW falling, prev >= 20": (d1 < 0) & (prev >= 20),
        "Signal 1992-2008": up & pdn & (prev >= 30) & (z.index < SPLIT),
        "Signal 2009-2026": up & pdn & (prev >= 30) & (z.index >= SPLIT),
    }
    rows = []
    for name, m in sets.items():
        row = {"Set": name}
        for lab, h in [("1m", 21), ("3m", 63), ("6m", 126)]:
            r = [e.px.iloc[i + h] / e.px.iloc[i] - 1
                 for i in e.rel_i[m.values] if i + h < e.n]
            row["N"] = len(r)
            row[f"{lab} avg %"] = np.mean(r) * 100
            row[f"{lab} hit %"] = np.mean(np.array(r) > 0) * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("Set")[
        ["N", "1m avg %", "1m hit %", "3m avg %", "3m hit %", "6m avg %", "6m hit %"]]


def grid(e: Engine) -> pd.DataFrame:
    rows = []
    for L, f, ex in itertools.product([0, 10, 20, 30, 40], ["none", "week", "month"],
                                      ["zew1", "zew2", "hold3", "hold6", "hold12"]):
        kw = dict(L=L, filt=f, exit="hold", hold=int(ex[4:])) if ex.startswith("hold") \
            else dict(L=L, filt=f, exit=ex)
        pos, tr = e.run(**kw)
        r = e.returns(pos)[START:]
        s = stats(r)
        rows.append({"Level": L, "Filter": f, "Exit": ex, **s, "Trades": len(tr),
                     "Sharpe 92-08": stats(r[:SPLIT])["Sharpe"],
                     "Sharpe 09-26": stats(r[SPLIT:])["Sharpe"]})
    return pd.DataFrame(rows)


def ism_combo(e: Engine, zew_pos: np.ndarray):
    """Union of the ISM-MACD regime (best prior strategy) and the ZEW signal."""
    import ism_macd_strategy as im
    ism = im.load_ism()
    hist = im.macd(ism, 6, 12, 5)[2]
    reg = pd.Series(im.ism_regime_raw(hist.values, 3, 5), index=ism.index)
    # ISM for month m is published on the 1st business day of m+1 -> act 3 days later
    reg.index = reg.index + pd.offsets.BDay(3)
    ism_pos = reg.reindex(e.idx, method="ffill").fillna(0).values
    both = np.maximum(ism_pos, zew_pos)
    lo, hi = "2000-06-01", str(ism.index[-1].date())
    out = {}
    for name, p in [("ISM MACD only", ism_pos), ("ZEW surprise only", zew_pos),
                    ("ISM OR ZEW", both)]:
        out[name] = stats(e.returns(p)[lo:hi])
    out["DAX buy & hold"] = stats(pd.Series(e.ret, index=e.idx)[lo:hi])
    return pd.DataFrame(out).T, (lo, hi), e.returns(both)[lo:hi]


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot(e, pos, trades, r, bh, path):
    fig, ax = plt.subplots(3, 1, figsize=(15, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 2]})
    S, B = "#1f9d55", "#2b6cb0"
    cs, cb = (1 + r).cumprod(), (1 + bh).cumprod()
    ax[0].plot(cb.index, cb, color=B, lw=1.5, label="DAX buy & hold")
    ax[0].plot(cs.index, cs, color=S, lw=2.2, label="ZEW surprise strategy")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)")
    ax[0].set_title("ZEW positive-surprise strategy on DAX — "
                    f"trough ≥ {BASE['L']}, prior month candle up, exit on next ZEW decline",
                    fontweight="bold")
    ax[1].fill_between(cb.index, (cb / cb.cummax() - 1) * 100, 0, color=B, alpha=.3, label="DAX")
    ax[1].fill_between(cs.index, (cs / cs.cummax() - 1) * 100, 0, color=S, alpha=.6, label="Strategy")
    ax[1].set_ylabel("Drawdown %")
    z = e.zew[START:]
    ax[2].plot(z.index, z, color="#dd6b20", lw=1.4, label="ZEW Economic Sentiment")
    ax[2].axhline(BASE["L"], color="grey", ls="--", lw=1, label=f"Level {BASE['L']}")
    ax[2].axhline(0, color="black", lw=.6)
    sig = [e.zew.index[k] for k in range(len(e.zew)) if e.signal(k, BASE["L"])]
    ax[2].scatter(sig, e.zew.loc[sig], marker="^", s=70, color=S, zorder=5, label="Signal")
    ax[2].set_ylabel("Balance")
    for a, b in trades:
        for x in ax:
            x.axvspan(e.idx[a], e.idx[b], color=S, alpha=.12, lw=0)
    for x in ax:
        x.grid(alpha=.25); x.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.2f}".format)
    e = Engine(load_zew(), load_dax())
    print(f"ZEW {e.zew.index[0].date()} -> {e.zew.index[-1].date()} ({len(e.zew)} releases) | "
          f"DAX {e.idx[0].date()} -> {e.idx[-1].date()}")

    print("\n== 1. EVENT STUDY: DAX forward return from the release-day close ==")
    print(event_study(e).to_string())

    pos, trades = e.run(**BASE)
    r = e.returns(pos)[START:]
    bh = pd.Series(e.ret, index=e.idx)[START:]
    print(f"\n== 2. BASE STRATEGY {BASE} ==")
    tab = pd.DataFrame({"Strategy": stats(r), "DAX B&H": stats(bh),
                        "Strategy 92-08": stats(r[:SPLIT]), "B&H 92-08": stats(bh[:SPLIT]),
                        "Strategy 09-26": stats(r[SPLIT:]), "B&H 09-26": stats(bh[SPLIT:])}).T
    print(tab.to_string())
    tr = pd.DataFrame([{"Entry": e.idx[a].date(), "Exit": e.idx[b].date(),
                        "Days": (e.idx[b] - e.idx[a]).days,
                        "Return %": (e.px.iloc[b] / e.px.iloc[a] - 1) * 100} for a, b in trades])
    print(f"\nTrades: {len(tr)} | win {(tr['Return %'] > 0).mean()*100:.0f}% | "
          f"avg {tr['Return %'].mean():.1f}% | worst {tr['Return %'].min():.1f}% | "
          f"best {tr['Return %'].max():.1f}%")
    print(tr.to_string(index=False))
    inv = (pos[e.idx >= START] > 0).mean()
    print(f"\nReturn while invested (annualised): {r[r != 0].mean() * 252 * 100:.1f}%  "
          f"vs DAX all days {bh.mean() * 252 * 100:.1f}%   (invested {inv*100:.0f}% of days)")

    print("\n== 3. ROBUSTNESS GRID (75 variants) ==")
    g = grid(e)
    g.to_csv(os.path.join(HERE, "zew_surprise_grid.csv"), index=False)
    for c in ["Level", "Filter", "Exit"]:
        print(g.groupby(c)[["CAGR %", "Sharpe", "MaxDD %", "Invested %", "Trades",
                            "Sharpe 92-08", "Sharpe 09-26"]].median().to_string(), "\n")
    print(f"Variants with Sharpe > B&H ({stats(bh)['Sharpe']:.2f}): "
          f"{(g['Sharpe'] > stats(bh)['Sharpe']).sum()} of {len(g)}")

    print("\n== 4. COMBINATION WITH ISM-MACD REGIME ==")
    combo, (lo, hi), _ = ism_combo(e, pos)
    print(f"Period {lo} -> {hi}")
    print(combo.to_string())

    plot(e, pos, trades, r, bh, os.path.join(HERE, "zew_surprise_backtest.png"))
    print("\nChart -> zew_surprise_backtest.png")


if __name__ == "__main__":
    main()
