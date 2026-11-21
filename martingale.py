"""
Martingale overlay on the Defensive + Growth combo  (DAX + Bund)
================================================================
Capital is treated as unlimited and external, so everything is measured in
EUR, not in %: a fixed base stake of BASE euros of notional is traded (no
compounding) and the martingale multiplies that stake.

P&L is futures-style: stake x (strategy return - cash rate), i.e. no interest
is credited on capital.  Costs 0.10% of traded notional, including the
notional added or removed when the stake changes while invested.

A) Drawdown martingale - the stake follows the drawdown of the 1x strategy:
     level = floor(drawdown / STEP);  stake = 2^level, 1.5^level or 1+level
     back to 1x when the 1x strategy makes a new high.
B) Trade martingale - classic: after every losing DAX trade the next trade's
     stake is multiplied by k; reset to 1x after a winning trade.
The stake decided at a close applies from the next day.
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


def eur_stats(pnl: pd.Series, stake: np.ndarray) -> dict:
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    y = pnl.groupby(pnl.index.year).sum()
    yrs = len(pnl) / 252
    under = (dd < 0).astype(int)
    return {"EUR / year (avg)": pnl.sum() / yrs, "Median year": y.median(), "Worst year": y.min(),
            "Best year": y.max(), "Losing years": int((y < 0).sum()),
            "Max EUR drawdown": dd.min(), "EUR per year / max DD": pnl.sum() / yrs / -dd.min(),
            "Longest DD (months)": under.groupby((under == 0).cumsum()).sum().max() / 21,
            "Avg stake (x base)": stake.mean(), "Max stake (x base)": stake.max(),
            "Max notional EUR": stake.max() * BASE, "Total EUR": pnl.sum()}


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:,.0f}".format)
    D = bc.build()
    idx, rd, rb, rf = D["idx"], D["rd"], D["rb"], D["rf"]
    n = len(idx)
    sub = {}
    for name, (z, m, op, sc, f, g, b) in PICKS.items():
        _, dpos, bpos = bc.simulate(D, bc.dax_decision(D, z, m, op, sc, f, g), D["B"][b])
        sub[name] = (dpos, bpos)

    def pnl_of(dpos, bpos, mult):
        """EUR P&L of weights (dpos, bpos) scaled by stake multiplier `mult` (already lagged)."""
        wd, wb = dpos * mult, bpos * mult
        hd, hb = np.roll(wd, 1), np.roll(wb, 1); hd[0] = hb[0] = 0
        turn = np.abs(np.diff(wd, prepend=0)) + np.abs(np.diff(wb, prepend=0))
        return BASE * (hd * (rd - rf) + hb * (rb - rf) - turn * COST)

    wd = (sub["Defensive"][0] + sub["Growth"][0]) / 2
    wb = (sub["Defensive"][1] + sub["Growth"][1]) / 2
    one = np.ones(n)
    base = pd.Series(pnl_of(wd, wb, one), index=idx)

    # ── A) drawdown martingale ───────────────────────────────────────────────
    bcum = base.cumsum().values
    bdd = (np.maximum.accumulate(bcum) - bcum) / BASE * 100        # drawdown of 1x, in % of base stake

    def dd_mult(step, kind, cap):
        lvl = np.floor(bdd / step)
        m = {"x2": 2.0 ** lvl, "x1.5": 1.5 ** lvl, "+1": 1 + lvl}[kind]
        m = np.minimum(m, cap)
        m = np.roll(m, 1); m[0] = 1
        return m

    # ── B) trade martingale (per sub-strategy, DAX leg) ──────────────────────
    def trade_mult(dpos, k, cap):
        ex = dpos_pnl = np.roll(dpos, 1) * (rd - rf)
        m, cur, losses, acc, intrade = np.ones(n), 1.0, 0, 0.0, False
        for i in range(1, n):
            if dpos[i - 1] > 0:
                acc += ex[i]
            if dpos[i] > 0 and not intrade:                       # new trade opens at this close
                intrade, acc = True, 0.0
                cur = min(k ** losses, cap)
            elif dpos[i] == 0 and intrade:                        # trade closed
                intrade = False
                losses = losses + 1 if acc < 0 else 0
            m[i] = cur
        return m

    rows, series = {}, {}

    def add(name, pnl, stake):
        s = pd.Series(pnl, index=idx)
        rows[name], series[name] = eur_stats(s, stake), (s, stake)

    add("Base 1x (no martingale)", base.values, one)
    for step, kind, cap in itertools.product([3, 5, 8], ["x2", "x1.5", "+1"], [4, 8, 1e9]):
        m = dd_mult(step, kind, cap)
        add(f"DD step {step}% | {kind:4s} | cap {'none' if cap > 1e6 else int(cap)}", pnl_of(wd, wb, m), m)
    for k, cap in itertools.product([1.5, 2.0], [4, 8, 1e9]):
        md, mg = trade_mult(sub["Defensive"][0], k, cap), trade_mult(sub["Growth"][0], k, cap)
        p = 0.5 * pnl_of(sub["Defensive"][0], sub["Defensive"][1], md) + 0.5 * pnl_of(sub["Growth"][0], sub["Growth"][1], mg)
        add(f"Trade x{k} after loss | cap {'none' if cap > 1e6 else int(cap)}", p, (md * sub['Defensive'][0] + mg * sub['Growth'][0]) / 2
            + (1 - (sub['Defensive'][0] + sub['Growth'][0]) / 2))
    T = pd.DataFrame(rows).T
    T.to_csv(os.path.join(HERE, "martingale_grid.csv"))
    b = T.iloc[0]
    # fair comparison: a FIXED stake scaled to the same max EUR drawdown
    T["Fixed stake w/ same max DD: EUR/yr"] = -T["Max EUR drawdown"] * b["EUR per year / max DD"]
    T["Martingale edge vs fixed, EUR/yr"] = T["EUR / year (avg)"] - T["Fixed stake w/ same max DD: EUR/yr"]

    print(f"Period {idx[0].date()} -> {idx[-1].date()} | base stake EUR {BASE:,} notional | P&L excludes interest on capital")
    cols = ["EUR / year (avg)", "Worst year", "Losing years", "Max EUR drawdown", "Longest DD (months)",
            "Max notional EUR", "Fixed stake w/ same max DD: EUR/yr", "Martingale edge vs fixed, EUR/yr"]
    pd.set_option("display.max_rows", 100)
    print(T[cols].to_string())
    print("\nRatio EUR per year / max EUR drawdown (higher = better), and stake multiples:")
    print(T[["EUR per year / max DD", "Avg stake (x base)", "Max stake (x base)"]].to_string(float_format="{:.2f}".format))

    pick = "DD step 5% | +1   | cap 4"
    print(f"\n== CHOSEN: {pick} ==")
    print(pd.DataFrame({"Base 1x": rows["Base 1x (no martingale)"], "Martingale": rows[pick]}).to_string(float_format="{:,.2f}".format))
    s, stake = series[pick]
    yr = pd.DataFrame({"Base 1x": base, "Martingale": s}).groupby(lambda d: d.year).sum()
    print("\nEUR per calendar year"); print(yr.T.to_string())
    print("\nShare of days at each stake: " + ", ".join(f"{int(v)}x {100 * (stake == v).mean():.0f}%" for v in np.unique(stake)))
    print(f"Today ({idx[-1].date()}): stake {stake[-1]:.0f}x  (1x drawdown now {bdd[-1]:.1f}% of base stake)")

    fig, ax = plt.subplots(3, 1, figsize=(15, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]})
    scale = rows[pick]["Max EUR drawdown"] / rows["Base 1x (no martingale)"]["Max EUR drawdown"]
    lines = [("Base 1x", base, "#a0aec0", 1.4), (f"Fixed {scale:.1f}x (same max EUR drawdown)", base * scale, "#c05621", 1.4),
             ("Martingale: +1 unit per 5% drawdown, max 4x", s, "#1a365d", 2.4)]
    for name, p, col, lw in lines:
        c = p.cumsum()
        ax[0].plot(idx, c / 1000, color=col, lw=lw, label=f"{name}:  EUR {c.iloc[-1]/1000:,.0f}k total")
        ax[1].plot(idx, (c - c.cummax()) / 1000, color=col, lw=1.2 if lw < 2 else 1.6)
    ax[0].set_ylabel("Cumulative P&L, EUR thousand"); ax[1].set_ylabel("EUR drawdown, thousand")
    ax[0].set_title(f"Martingale on the Defensive + Growth combo — EUR P&L on a EUR {BASE/1000:.0f}k base stake", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=10)
    ax[2].fill_between(idx, stake, step="post", color="#1a365d", alpha=.6)
    ax[2].set_ylabel("Stake (x base)"); ax[2].set_yticks([1, 2, 3, 4])
    for a in ax:
        a.grid(alpha=.25); a.set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "martingale.png"), dpi=130, facecolor="white")
    print("\nChart -> martingale.png")


if __name__ == "__main__":
    main()
