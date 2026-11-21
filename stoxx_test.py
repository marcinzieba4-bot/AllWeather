"""
Euro Stoxx as the equity leg: Defensive / Growth / combo on Stoxx vs DAX
========================================================================
Same rules as combo_portfolio.py; only the equity index changes.

Stoxx = euro-zone large-cap total return in EUR, spliced from ETFs:
  2000-08..2007 iShares MSCI Eurozone (EZU) converted USD->EUR (FRED DEXUSEU)
  2008..        iShares Core EURO STOXX 50 (EXW1.DE)
(data/stoxx_tr.csv; index-provider history was not accessible.)

"own" signals  : candle stop and monthly MACD computed on the traded index
"DAX" signals  : computed on the DAX, only the traded instrument is Stoxx
The ZEW regime and the Bund rule are identical in all variants.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import best_combo_search as bc
import zew_bund_strategy as zb
import zew_regime_strategy as zr
from combo_portfolio import risk_stats
from zew_surprise_strategy import load_zew, load_dax, COST, HERE

SPLIT = "2012-01-01"


def main():
    pd.set_option("display.width", 230); pd.set_option("display.float_format", "{:.2f}".format)
    zew, dax, bund = load_zew(), load_dax(), zb.load_bund()
    stoxx = pd.read_csv(os.path.join(HERE, "data", "stoxx_tr.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    idx = stoxx.index.intersection(bund.index).intersection(dax.index)
    idx = idx[idx >= "2001-01-01"]                      # leaves warm-up for Stoxx candles
    px = {"DAX": dax, "Stoxx": stoxx}
    ret = {k: v.pct_change().reindex(idx).fillna(0).values for k, v in px.items()}
    rb = bund.pct_change().reindex(idx).fillna(0).values

    def al(arr, src):
        return pd.Series(np.asarray(arr, float), index=src).reindex(idx, method="ffill").fillna(0).values > 0

    sig = {}
    for k, p in px.items():
        lab = zr.Lab(zew, p)
        zleg = lab.regime() & lab.mom["m_SL2_RE1"]
        sig[k] = {"Defensive": al(zleg, lab.idx),
                  "Growth": al(zleg | bc.macd_rising(p, "M"), lab.idx)}
        rf = pd.Series(lab.rf, index=lab.idx).reindex(idx).fillna(0).values
    LB = zr.Lab(zew, bund)
    bdec = al(zb.falling_regime(LB.e), LB.idx)
    rfs = pd.Series(rf, index=idx)

    def run(style, traded, signals):
        dpos = np.roll(sig[signals][style], 1).astype(float); dpos[0] = 0
        bpos = np.roll(bdec, 1).astype(float); bpos[0] = 0
        bpos *= 1 - dpos
        hd, hb = np.roll(dpos, 1), np.roll(bpos, 1); hd[0] = hb[0] = 0
        turn = np.abs(np.diff(dpos, prepend=0)) + np.abs(np.diff(bpos, prepend=0))
        return pd.Series(hd * ret[traded] + hb * rb + (1 - hd - hb) * rf - turn * COST, index=idx), dpos

    keep = ["CAGR %", "Volatility %", "Sharpe", "Max drawdown %", "Worst year %", "Losing years"]

    def table(d):
        t = pd.DataFrame({k: risk_stats(v, rfs) for k, v in d.items()}).T[keep]
        for lab_, sl in [("01-11", slice(None, "2011-12-31")), ("12-26", slice(SPLIT, None))]:
            t[f"CAGR {lab_}"] = [risk_stats(v[sl], rfs[sl])["CAGR %"] for v in d.values()]
            t[f"Sharpe {lab_}"] = [risk_stats(v[sl], rfs[sl])["Sharpe"] for v in d.values()]
        return t

    print(f"Period {idx[0].date()} -> {idx[-1].date()} | daily correlation DAX vs Stoxx "
          f"{np.corrcoef(ret['DAX'], ret['Stoxx'])[0, 1]:.2f}")
    print("\n== 1. BUY & HOLD ==")
    bh = {"DAX buy & hold": pd.Series(ret["DAX"], index=idx), "Stoxx buy & hold": pd.Series(ret["Stoxx"], index=idx)}
    print(table(bh).to_string())

    print("\n== 2. EACH RULE SET ON EACH INDEX ==")
    R = {}
    for style in ["Defensive", "Growth"]:
        R[f"{style} | DAX"] = run(style, "DAX", "DAX")[0]
        R[f"{style} | Stoxx, own signals"] = run(style, "Stoxx", "Stoxx")[0]
        R[f"{style} | Stoxx, DAX signals"] = run(style, "Stoxx", "DAX")[0]
    print(table(R).to_string())
    print("Time in equities: " + ", ".join(
        f"{s} DAX {run(s, 'DAX', 'DAX')[1].mean()*100:.0f}% / Stoxx-own {run(s, 'Stoxx', 'Stoxx')[1].mean()*100:.0f}%"
        for s in ["Defensive", "Growth"]))

    print("\n== 3. 50/50 COMBOS (Defensive leg + Growth leg) ==")
    C = {"Defensive DAX + Growth DAX": (R["Defensive | DAX"] + R["Growth | DAX"]) / 2,
         "Defensive Stoxx + Growth DAX   <- your idea": (R["Defensive | Stoxx, own signals"] + R["Growth | DAX"]) / 2,
         "Defensive DAX + Growth Stoxx": (R["Defensive | DAX"] + R["Growth | Stoxx, own signals"]) / 2,
         "Defensive Stoxx + Growth Stoxx": (R["Defensive | Stoxx, own signals"] + R["Growth | Stoxx, own signals"]) / 2,
         "Def. Stoxx (DAX signals) + Growth DAX": (R["Defensive | Stoxx, DAX signals"] + R["Growth | DAX"]) / 2,
         "Half DAX / half Stoxx in both legs": (R["Defensive | DAX"] + R["Growth | DAX"]
                                                + R["Defensive | Stoxx, own signals"] + R["Growth | Stoxx, own signals"]) / 4}
    print(table(C).to_string())
    yr = ((1 + pd.DataFrame({"DAX/DAX combo": C["Defensive DAX + Growth DAX"],
                             "Stoxx/DAX combo": C["Defensive Stoxx + Growth DAX   <- your idea"],
                             "DAX B&H": bh["DAX buy & hold"], "Stoxx B&H": bh["Stoxx buy & hold"]})
           ).groupby(lambda d: d.year).prod() - 1) * 100
    print("\nCalendar years %"); print(yr.T.to_string())

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    lines = [("DAX buy & hold", bh["DAX buy & hold"], "#a0aec0", 1.2, "-"),
             ("Stoxx buy & hold", bh["Stoxx buy & hold"], "#a0aec0", 1.2, "--"),
             ("Combo: Defensive DAX + Growth DAX", C["Defensive DAX + Growth DAX"], "#1a365d", 2.4, "-"),
             ("Combo: Defensive Stoxx + Growth DAX", C["Defensive Stoxx + Growth DAX   <- your idea"], "#c05621", 2.0, "-"),
             ("Combo: Defensive Stoxx + Growth Stoxx", C["Defensive Stoxx + Growth Stoxx"], "#2f855a", 1.4, "-")]
    for name, r, col, lw, ls in lines:
        c = (1 + r).cumprod()
        ax[0].plot(c.index, c, color=col, lw=lw, ls=ls, label=f"{name}  (x{c.iloc[-1]:.1f})")
        ax[1].plot(c.index, (c / c.cummax() - 1) * 100, color=col, lw=1.1, ls=ls)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)"); ax[1].set_ylabel("Drawdown %")
    ax[0].set_title("Stoxx vs DAX as the equity leg of the combined strategy", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    for a in ax:
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "stoxx_test.png"), dpi=130, facecolor="white")
    print("\nChart -> stoxx_test.png")


if __name__ == "__main__":
    main()
