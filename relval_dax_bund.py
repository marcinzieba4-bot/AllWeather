"""
DAX vs Bund relative value and 60/40, driven by the Defensive + Growth signals
==============================================================================
Signal s  = combo DAX weight (0, 0.5 or 1: Defensive and Growth averaged)
Signal b  = combo Bund weight (ZEW has fallen twice in a row while >= 0, and DAX leg out)

Structures (weights as multiples of the base stake):
  Combo            DAX s,            Bund b            (the existing strategy)
  DAX only         DAX s
  RelVal 1x        DAX +s,           Bund -s           long DAX / short Bund, equal notional
  RelVal 3x        DAX +s,           Bund -3s          Bund leg sized to similar risk
  RelVal 2-sided   DAX +s-b,         Bund -(s-b) x k   reversed (long Bund / short DAX) on the Bund signal
  60/40 on signal  DAX 0.6s,         Bund 0.4s         (+ Bund b when DAX leg is out, in one variant)
  DAX + Bund       DAX s,            Bund +s           both long on the equity signal

Stop-loss: if the structure's P&L since entry falls to -X% of the base stake,
go flat at the next close.  Re-entry either on a fresh signal (signal goes to
zero and comes back) or after the next up month in the DAX.

EUR per 100k base stake, futures-style (no interest on capital), 0.10% of
traded notional per leg.  Daily closes, next-close execution.
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import best_combo_search as bc
from combo_portfolio import PICKS
from zew_surprise_strategy import COST, HERE

BASE = 100_000


def run(D, twd, twb, stop=None, reentry="fresh", up_month=None):
    """twd/twb: target weights held from close i to i+1.  Returns EUR P&L series and actual weights."""
    rd, rb, rf = D["rd"] - D["rf"], D["rb"] - D["rf"], D["rf"]
    n = len(rd)
    awd, awb, pnl = np.zeros(n), np.zeros(n), np.zeros(n)
    stopped = pend_stop = False
    cum, stops = 0.0, 0
    for i in range(1, n):
        g = awd[i - 1] * rd[i] + awb[i - 1] * rb[i]
        cum += g
        flat_target = twd[i] == 0 and twb[i] == 0
        if pend_stop:
            stopped, pend_stop = True, False
            stops += 1
        if stopped:
            if flat_target or (reentry == "up month" and up_month[i] and not up_month[i - 1]):
                stopped = False
        wd, wb = (0.0, 0.0) if stopped else (twd[i], twb[i])
        if awd[i - 1] == 0 and awb[i - 1] == 0 or np.sign(wd) != np.sign(awd[i - 1]) and awd[i - 1] != 0:
            cum = 0.0                                              # new trade (or direction flip)
        if stop is not None and (wd or wb) and cum <= -stop:
            pend_stop = True
        awd[i], awb[i] = wd, wb
        pnl[i] = g - COST * (abs(wd - awd[i - 1]) + abs(wb - awb[i - 1]))
    return pd.Series(pnl * BASE, index=D["idx"]), awd, awb, stops


def eur(p, awd, awb, stops=0):
    c = p.cumsum()
    dd = c - c.cummax()
    y = p.groupby(p.index.year).sum()
    under = (dd < 0).astype(int)
    yrs = len(p) / 252
    return {"EUR / year": p.sum() / yrs, "Max EUR DD": dd.min(), "EUR/yr per EUR of DD": p.sum() / yrs / -dd.min(),
            "Worst year": y.min(), "Losing years": int((y < 0).sum()), "Sharpe": p.mean() / p.std() * np.sqrt(252),
            "Longest DD (mo)": under.groupby((under == 0).cumsum()).sum().max() / 21,
            "Gross notional avg (x)": (np.abs(awd) + np.abs(awb)).mean(), "Stops hit": stops}


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:,.0f}".format); pd.set_option("display.max_rows", 200)
    D = bc.build()
    idx, n = D["idx"], len(D["idx"])
    w = {}
    for name, (z, m, op, sc, f, g, b) in PICKS.items():
        _, dpos, bpos = bc.simulate(D, bc.dax_decision(D, z, m, op, sc, f, g), D["B"][b])
        w[name] = (dpos, bpos)
    s = (w["Defensive"][0] + w["Growth"][0]) / 2
    b = (w["Defensive"][1] + w["Growth"][1]) / 2
    up = np.roll(D["F"]["m_up1"], 1)
    S = {"Combo: DAX, else Bund rule": (s, b),
         "DAX only": (s, 0 * s),
         "RelVal 1x: long DAX / short Bund": (s, -s),
         "RelVal 3x Bund": (s, -3 * s),
         "RelVal 2-sided 1x": (s - b, b - s),
         "RelVal 2-sided 3x Bund": (s - b, 3 * (b - s)),
         "60/40 on signal, else flat": (0.6 * s, 0.4 * s),
         "60/40 on signal, else Bund rule": (0.6 * s, 0.4 * s + b),
         "DAX + Bund 1:1 on signal, else Bund rule": (s, s + b),
         "Static 60/40 (always on)": (0.6 + 0 * s, 0.4 + 0 * s),
         "DAX buy & hold": (1 + 0 * s, 0 * s), "Bund buy & hold": (0 * s, 1 + 0 * s)}

    print(f"Period {idx[0].date()} -> {idx[-1].date()} | EUR per {BASE:,} base stake | "
          f"correlation DAX vs Bund daily {np.corrcoef(D['rd'], D['rb'])[0, 1]:.2f}")
    sig = s > 0
    print(f"While the DAX signal is on ({sig.mean()*100:.0f}% of days): DAX {252*100*(D['rd']-D['rf'])[1:][sig[:-1]].mean():.1f}% a year over cash, "
          f"Bund {252*100*(D['rb']-D['rf'])[1:][sig[:-1]].mean():.1f}%.  While off: DAX "
          f"{252*100*(D['rd']-D['rf'])[1:][~sig[:-1]].mean():.1f}%, Bund {252*100*(D['rb']-D['rf'])[1:][~sig[:-1]].mean():.1f}%")

    print("\n== 1. STRUCTURES WITHOUT EXTRA STOP-LOSS ==")
    res, keep = {}, {}
    for k, (twd, twb) in S.items():
        p, awd, awb, _ = run(D, twd, twb)
        res[k], keep[k] = eur(p, awd, awb), p
    print(pd.DataFrame(res).T.to_string(formatters={"EUR/yr per EUR of DD": "{:.2f}".format, "Sharpe": "{:.2f}".format,
                                                    "Gross notional avg (x)": "{:.2f}".format}))

    print("\n== 2. WITH STOP-LOSSES (stop in % of base stake since entry) ==")
    rows = {}
    core = ["Combo: DAX, else Bund rule", "DAX only", "RelVal 1x: long DAX / short Bund", "RelVal 3x Bund",
            "RelVal 2-sided 1x", "60/40 on signal, else flat", "60/40 on signal, else Bund rule", "DAX + Bund 1:1 on signal, else Bund rule"]
    for k, stop, re in itertools.product(core, [None, 0.05, 0.075, 0.10], ["fresh", "up month"]):
        if stop is None and re == "up month":
            continue
        p, awd, awb, st = run(D, *S[k], stop=stop, reentry=re, up_month=up)
        tag = "no stop" if stop is None else f"stop {stop*100:.1f}% | re-entry {re}"
        rows[(k, tag)] = eur(p, awd, awb, st)
        keep[(k, tag)] = p
    T = pd.DataFrame(rows).T
    T.to_csv(os.path.join(HERE, "relval_dax_bund_grid.csv"))
    print(T[["EUR / year", "Max EUR DD", "EUR/yr per EUR of DD", "Worst year", "Losing years", "Longest DD (mo)", "Stops hit"]].to_string(
        formatters={"EUR/yr per EUR of DD": "{:.2f}".format}))

    print("\n== 3. EFFECT OF STOPS, averaged over the 8 structures ==")
    A = T.reset_index().rename(columns={"level_0": "structure", "level_1": "stop"})
    print(A.groupby("stop")[["EUR / year", "Max EUR DD", "EUR/yr per EUR of DD", "Stops hit"]].mean().to_string(
        formatters={"EUR/yr per EUR of DD": "{:.2f}".format}))

    yr = pd.DataFrame({k: keep[k].groupby(keep[k].index.year).sum() for k in
                       ["Combo: DAX, else Bund rule", "RelVal 1x: long DAX / short Bund", "60/40 on signal, else Bund rule",
                        "DAX + Bund 1:1 on signal, else Bund rule", "Static 60/40 (always on)"]})
    print("\n== 4. EUR PER CALENDAR YEAR =="); print(yr.T.to_string())

    fig, ax = plt.subplots(2, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    lines = [("Static 60/40 (always on)", "#a0aec0", 1.2), ("RelVal 1x: long DAX / short Bund", "#c53030", 1.5),
             ("RelVal 2-sided 1x", "#dd6b20", 1.2), ("60/40 on signal, else Bund rule", "#2f855a", 1.6),
             ("Combo: DAX, else Bund rule", "#1a365d", 2.4), ("DAX + Bund 1:1 on signal, else Bund rule", "#6b46c1", 1.6)]
    for k, col, lw in lines:
        c = keep[k].cumsum() / 1000
        ax[0].plot(idx, c, color=col, lw=lw, label=f"{k}:  EUR {c.iloc[-1]:,.0f}k")
        ax[1].plot(idx, c - c.cummax(), color=col, lw=1.1)
    ax[0].set_ylabel("Cumulative P&L, EUR thousand"); ax[1].set_ylabel("EUR drawdown, thousand")
    ax[0].set_title(f"DAX / Bund structures on the Defensive + Growth signals — EUR {BASE/1000:.0f}k base stake", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    for a in ax:
        a.grid(alpha=.25); a.set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "relval_dax_bund.png"), dpi=130, facecolor="white")
    print("\nChart -> relval_dax_bund.png")


if __name__ == "__main__":
    main()
