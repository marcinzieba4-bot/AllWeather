"""
Daily moving-average filters (50 to 100 days) on DAX and Bund
=============================================================
Filters, each computed on the asset it is applied to (known at the close):
  P>MA50 | P>MA75 | P>MA100   price above its simple moving average
  MA50>MA100                  50-day average above the 100-day average

Where they are applied to the Defensive + Growth combo:
  A  extra filter on the DAX leg         (DAX only held if DAX passes)
  B  extra filter on the Bund leg        (Bund only held if Bund passes)
  C  both
  D  MA filter REPLACES the monthly candle stop in the ZEW leg
  E  Bund leg = "Bund above its MA whenever out of the DAX" (replaces the ZEW Bund rule)
  F  the filters on their own: buy & hold DAX / Bund only while above the MA

EUR per 100k base stake, futures-style (no interest on capital), 0.10% of
traded notional, next-close execution.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import best_combo_search as bc
import zew_bund_strategy as zb
from combo_portfolio import PICKS
from relval_dax_bund import eur, BASE
from zew_surprise_strategy import load_dax, COST, HERE

SPLIT = "2012-01-01"


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:,.0f}".format); pd.set_option("display.max_rows", 200)
    D = bc.build()
    idx, n = D["idx"], len(D["idx"])
    rd, rb = D["rd"] - D["rf"], D["rb"] - D["rf"]

    def filters(px):
        m = {w: px.rolling(w).mean() for w in (50, 75, 100)}
        f = {"P>MA50": px > m[50], "P>MA75": px > m[75], "P>MA100": px > m[100], "MA50>MA100": m[50] > m[100]}
        return {k: v.reindex(idx, method="ffill").fillna(False).values for k, v in f.items()}

    FD, FB = filters(load_dax()), filters(zb.load_bund())
    ones, zeros = np.ones(n, bool), np.zeros(n, bool)
    dec = {k: bc.dax_decision(D, z, m, op, sc, f, g) for k, (z, m, op, sc, f, g, b) in PICKS.items()}
    zreg, macd = D["Z"]["cross_or_surprise"], D["M"]["macd_M"]
    bund_rule = D["B"]["zew_fall2_pos"]

    def pos(d):
        p = np.roll(d, 1).astype(float); p[0] = 0
        return p

    def combo(dD, dG, bdec):
        """50/50 of Defensive and Growth; Bund held by each half only when its DAX leg is out."""
        pD, pG, pB = pos(dD), pos(dG), pos(bdec)
        wd = (pD + pG) / 2
        wb = (pB * (1 - pD) + pB * (1 - pG)) / 2
        return wd, wb

    def pnl(wd, wb):
        hd, hb = np.roll(wd, 1), np.roll(wb, 1); hd[0] = hb[0] = 0
        turn = np.abs(np.diff(wd, prepend=0)) + np.abs(np.diff(wb, prepend=0))
        return pd.Series(BASE * (hd * rd + hb * rb - turn * COST), index=idx)

    rows, keep = {}, {}

    def add(name, wd, wb):
        p = pnl(wd, wb)
        s = eur(p, wd, wb)
        h1, h2 = p[:SPLIT], p[SPLIT:]
        dd = lambda q: (q.cumsum() - q.cumsum().cummax()).min()
        rows[name] = {"EUR / year": s["EUR / year"], "Max EUR DD": s["Max EUR DD"], "EUR/yr per EUR DD": s["EUR/yr per EUR of DD"],
                      "Worst year": s["Worst year"], "Losing years": s["Losing years"], "Sharpe": s["Sharpe"],
                      "Longest DD (mo)": s["Longest DD (mo)"],
                      "EUR/yr 97-11": h1.sum() / (len(h1) / 252), "EUR/yr 12-26": h2.sum() / (len(h2) / 252), "Max DD 12-26": dd(h2),
                      "DAX %": np.abs(wd).mean() * 100, "Bund %": np.abs(wb).mean() * 100,
                      "Turnover x/yr": (np.abs(np.diff(wd)).sum() + np.abs(np.diff(wb)).sum()) / (n / 252)}
        keep[name] = p

    add("BASELINE combo (no MA filter)", *combo(dec["Defensive"], dec["Growth"], bund_rule))
    for k in FD:
        add(f"A  DAX leg & DAX {k}", *combo(dec["Defensive"] & FD[k], dec["Growth"] & FD[k], bund_rule))
    for k in FB:
        add(f"B  Bund leg & Bund {k}", *combo(dec["Defensive"], dec["Growth"], bund_rule & FB[k]))
    for k in FD:
        add(f"C  both legs & {k}", *combo(dec["Defensive"] & FD[k], dec["Growth"] & FD[k], bund_rule & FB[k]))
    for k in FD:
        add(f"D  ZEW leg: DAX {k} replaces candle stop", *combo(zreg & FD[k], (zreg & FD[k]) | macd, bund_rule))
    for k in FB:
        add(f"E  Bund whenever out of DAX & Bund {k}", *combo(dec["Defensive"], dec["Growth"], FB[k]))
    add("F  DAX buy & hold", ones.astype(float), zeros.astype(float))
    for k in FD:
        add(f"F  DAX only while DAX {k}", pos(FD[k]), zeros.astype(float))
    add("F  Bund buy & hold", zeros.astype(float), ones.astype(float))
    for k in FB:
        add(f"F  Bund only while Bund {k}", zeros.astype(float), pos(FB[k]))

    T = pd.DataFrame(rows).T
    T.to_csv(os.path.join(HERE, "ma_filter_grid.csv"))
    print(f"Period {idx[0].date()} -> {idx[-1].date()} | EUR per {BASE:,} base stake")
    fm = {"EUR/yr per EUR DD": "{:.2f}".format, "Sharpe": "{:.2f}".format, "Turnover x/yr": "{:.1f}".format}
    print(T.to_string(formatters=fm))

    base = T.iloc[0]
    print("\n== SUMMARY BY PLACEMENT (average of the four MA variants, change vs baseline) ==")
    G = T.iloc[1:21].copy()
    G["group"] = [i[0] for i in G.index]
    S = G.groupby("group")[["EUR / year", "Max EUR DD", "EUR/yr per EUR DD", "EUR/yr 12-26", "Turnover x/yr"]].mean()
    S["EUR / year vs base"] = S["EUR / year"] - base["EUR / year"]
    S["Max DD vs base"] = S["Max EUR DD"] - base["Max EUR DD"]
    print(S.to_string(formatters=fm))
    better = T.iloc[1:21][(T.iloc[1:21]["EUR/yr per EUR DD"] > base["EUR/yr per EUR DD"])]
    print(f"\nVariants A-E beating the baseline ratio ({base['EUR/yr per EUR DD']:.2f}): {len(better)} of 20"
          + (": " + "; ".join(better.index) if len(better) else ""))
    both = T.iloc[1:21][(T.iloc[1:21]["EUR / year"] > base["EUR / year"]) & (T.iloc[1:21]["Max EUR DD"] > base["Max EUR DD"])]
    print(f"Variants with BOTH more EUR/year and a smaller max drawdown: {len(both)}" + (": " + "; ".join(both.index) if len(both) else ""))

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    lines = [("F  DAX only while DAX P>MA100", "#a0aec0", 1.2), ("A  DAX leg & DAX P>MA100", "#c53030", 1.4),
             ("D  ZEW leg: DAX P>MA100 replaces candle stop", "#dd6b20", 1.4), ("B  Bund leg & Bund P>MA100", "#2f855a", 1.6),
             ("E  Bund whenever out of DAX & Bund P>MA100", "#6b46c1", 1.4), ("BASELINE combo (no MA filter)", "#1a365d", 2.4)]
    for k, col, lw in lines:
        c = keep[k].cumsum() / 1000
        ax[0].plot(idx, c, color=col, lw=lw, label=f"{k}:  EUR {c.iloc[-1]:,.0f}k")
        ax[1].plot(idx, c - c.cummax(), color=col, lw=1.1)
    ax[0].set_ylabel("Cumulative P&L, EUR thousand"); ax[1].set_ylabel("EUR drawdown, thousand")
    ax[0].set_title(f"100-day moving-average filter on the combo — EUR {BASE/1000:.0f}k base stake", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    for a in ax:
        a.grid(alpha=.25); a.set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ma_filter_test.png"), dpi=130, facecolor="white")
    print("\nChart -> ma_filter_test.png")


if __name__ == "__main__":
    main()
