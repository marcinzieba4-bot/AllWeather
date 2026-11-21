"""
Relative value: SHORT DAX / LONG S&P 500 when ZEW crosses below zero
====================================================================
Entry  : ZEW Economic Sentiment crosses from >= 0 to < 0 (monthly release);
         both legs opened at the next close.
Exit   : take profit when the spread P&L since entry reaches TP (5 / 7.5 / 10%),
         else  a) ZEW back above 0,  b) time stop,  optionally  c) stop loss.
Spread : daily return of S&P 500 TR (USD) minus DAX TR (EUR) x hedge ratio.
         Futures-style and currency-hedged: each leg earns its local-currency
         return, no interest on capital.  Costs 0.10% per leg per side.
EUR figures are per EUR 100k on EACH leg.  Exits are checked on daily closes.
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zew_surprise_strategy import load_zew, load_dax, HERE

STAKE, COST, START = 100_000, 0.001, "1992-01-01"


def load():
    zew, dax = load_zew(), load_dax()
    spx = pd.read_csv(os.path.join(HERE, "data", "spx_tr.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    idx = dax.index.intersection(spx.index)
    idx = idx[idx >= START]
    return zew, dax.reindex(idx).pct_change().fillna(0), spx.reindex(idx).pct_change().fillna(0), idx


def simulate(zew, rd, rs, idx, tp=0.075, sl=None, max_months=None, zew_exit=True, hedge=1.0, direction=-1):
    """direction -1: enter on cross BELOW zero (short DAX / long SPX)."""
    spread = (rs - hedge * rd).values
    n, zv = len(idx), zew.values
    rel = idx.searchsorted(zew.index)
    cross_dn = {rel[k]: True for k in range(1, len(zv)) if zv[k] < 0 <= zv[k - 1] and rel[k] < n}
    back_up = {rel[k]: True for k in range(1, len(zv)) if zv[k] >= 0 and rel[k] < n}
    pnl, pos, trades = np.zeros(n), 0, []
    cum = entry = 0
    pend = None
    for i in range(1, n):
        if pos:
            pnl[i] += spread[i]
            cum = (1 + cum) * (1 + spread[i]) - 1
        if pend == "in":
            pos, cum, entry = 1, 0.0, i
            pnl[i] -= COST * (1 + hedge)
        elif pend:
            pnl[i] -= COST * (1 + hedge)
            trades.append({"Entry": idx[entry].date(), "Exit": idx[i].date(), "Months": (idx[i] - idx[entry]).days / 30.44,
                           "Spread %": cum * 100, "Reason": pend})
            pos = 0
        pend = None
        if pos:
            if cum >= tp:
                pend = "take profit"
            elif sl is not None and cum <= -sl:
                pend = "stop loss"
            elif zew_exit and back_up.get(i):
                pend = "ZEW back above 0"
            elif max_months and (idx[i] - idx[entry]).days / 30.44 >= max_months:
                pend = "time stop"
        elif cross_dn.get(i):
            pend = "in"
    if pos:
        trades.append({"Entry": idx[entry].date(), "Exit": "open", "Months": (idx[-1] - idx[entry]).days / 30.44,
                       "Spread %": cum * 100, "Reason": "open"})
    return pd.Series(pnl * STAKE, index=idx), pd.DataFrame(trades)


def eur(p, t):
    c = p.cumsum()
    dd = c - c.cummax()
    y = p.groupby(p.index.year).sum()
    done = t[t["Reason"] != "open"]
    return {"EUR / year": p.sum() / (len(p) / 252), "Max EUR drawdown": dd.min(), "Worst year": y.min(),
            "Trades": len(t), "Win %": (done["Spread %"] > 0).mean() * 100, "TP hit %": (done["Reason"] == "take profit").mean() * 100,
            "Avg trade %": done["Spread %"].mean(), "Worst trade %": done["Spread %"].min(),
            "Avg months": done["Months"].mean(), "In trade %": (p != 0).mean() * 100, "Total EUR": p.sum()}


def main():
    pd.set_option("display.width", 230); pd.set_option("display.float_format", "{:,.1f}".format)
    zew, rd, rs, idx = load()
    spread = rs - rd
    sp = (1 + spread).cumprod()
    print(f"Period {idx[0].date()} -> {idx[-1].date()}")
    print(f"Always long SPX / short DAX: {((sp.iloc[-1]) ** (252 / len(sp)) - 1) * 100:.2f}% a year, "
          f"vol {spread.std() * np.sqrt(252) * 100:.1f}%, DAX beta to SPX {np.cov(rd, rs)[0, 1] / rs.var():.2f}")

    print("\n== 1. EVENT STUDY: SPX minus DAX after ZEW releases (avg % / share positive) ==")
    prev = zew.shift(1)
    rel = idx.searchsorted(zew.index)
    sets = {"All releases": pd.Series(True, index=zew.index), "Cross BELOW 0  <- signal": (zew < 0) & (prev >= 0),
            "Cross above 0": (zew >= 0) & (prev < 0), "ZEW < 0": zew < 0, "ZEW >= 0": zew >= 0}
    rows = []
    for name, m in sets.items():
        row = {"Set": name}
        for lab, h in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
            r = np.array([sp.iloc[i + h] / sp.iloc[i] - 1 for i, ok, d in zip(rel, m.values, zew.index)
                          if ok and d >= idx[0] and i + h < len(idx)])
            row["N"] = len(r); row[lab] = f"{r.mean() * 100:5.1f} / {(r > 0).mean() * 100:3.0f}%"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Set").to_string())

    print("\n== 2. HOW FAR DOES THE SPREAD RUN AFTER EACH CROSS BELOW ZERO? (held until ZEW is back above 0) ==")
    _, t0 = simulate(zew, rd, rs, idx, tp=9.9)
    path = []
    for e, x in zip(t0["Entry"], t0["Exit"]):
        a = idx.get_loc(pd.Timestamp(e)); b = len(idx) - 1 if x == "open" else idx.get_loc(pd.Timestamp(x))
        seg = (1 + spread.iloc[a + 1:b + 1]).cumprod() - 1
        path.append({"Entry": e, "Exit": x, "Months": (idx[b] - idx[a]).days / 30.44, "Best %": max(seg.max() * 100, 0),
                     "Worst %": min(seg.min() * 100, 0), "Final %": seg.iloc[-1] * 100})
    path = pd.DataFrame(path)
    print(path.to_string(index=False))
    print(f"Reached +5%: {(path['Best %'] >= 5).sum()} of {len(path)} | +7.5%: {(path['Best %'] >= 7.5).sum()} | "
          f"+10%: {(path['Best %'] >= 10).sum()} | fell below -10% at some point: {(path['Worst %'] <= -10).sum()}")

    print("\n== 3. STRATEGY VARIANTS, EUR per 100k per leg ==")
    res, keep = {}, {}
    for tp, (ex_name, kw) in itertools.product([0.05, 0.075, 0.10], [
            ("until ZEW > 0", dict()), ("until ZEW > 0, stop -10%", dict(sl=0.10)),
            ("max 6 months", dict(zew_exit=False, max_months=6)), ("max 12 months", dict(zew_exit=False, max_months=12)),
            ("max 12 months, stop -10%", dict(zew_exit=False, max_months=12, sl=0.10))]):
        p, t = simulate(zew, rd, rs, idx, tp=tp, **kw)
        k = f"TP {tp * 100:.1f}% | {ex_name}"
        res[k], keep[k] = eur(p, t), (p, t)
    p, t = simulate(zew, rd, rs, idx, tp=9.9); res["No take profit | until ZEW > 0"] = eur(p, t); keep["none"] = (p, t)
    p, t = simulate(zew, rd, rs, idx, tp=0.075, hedge=0.8); res["TP 7.5% | until ZEW > 0 | DAX leg 0.8x (beta-adjusted)"] = eur(p, t)
    T = pd.DataFrame(res).T
    T.to_csv(os.path.join(HERE, "relval_dax_spx_grid.csv"))
    print(T.to_string())

    pick = "TP 7.5% | until ZEW > 0"
    p, t = keep[pick]
    print(f"\n== 4. TRADES: {pick} ==")
    print(t.to_string(index=False))
    for lab, sl in [("1992-2008", slice(None, "2008-12-31")), ("2009-2026", slice("2009-01-01", None))]:
        q = p[sl]; c = q.cumsum()
        print(f"{lab}: EUR {q.sum() / (len(q) / 252):,.0f} a year, max EUR drawdown {(c - c.cummax()).min():,.0f}")

    fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6, 1.2]})
    for k, col, lw in [("TP 5.0% | until ZEW > 0", "#2f855a", 1.4), (pick, "#1a365d", 2.3), ("TP 10.0% | until ZEW > 0", "#c05621", 1.4), ("none", "#a0aec0", 1.2)]:
        c = keep[k][0].cumsum() / 1000
        ax[0].plot(idx, c, color=col, lw=lw, label=f"{'No take profit' if k == 'none' else k.split(' |')[0]}:  EUR {c.iloc[-1]:,.0f}k")
    ax[0].set_ylabel("Cumulative P&L, EUR thousand"); ax[0].legend(loc="upper left", fontsize=10)
    ax[0].set_title("Short DAX / long S&P 500 after ZEW crosses below zero — EUR 100k per leg, exit when ZEW is back above 0", fontweight="bold")
    ax[1].plot(idx, sp, color="#4a5568", lw=1.2); ax[1].set_yscale("log"); ax[1].set_ylabel("S&P 500 / DAX (total return, log)")
    ax[2].plot(zew[START:].index, zew[START:], color="#dd6b20", lw=1.2); ax[2].axhline(0, color="black", lw=.8); ax[2].set_ylabel("ZEW")
    for _, r in t.iterrows():
        a = pd.Timestamp(r["Entry"]); b = idx[-1] if r["Exit"] == "open" else pd.Timestamp(r["Exit"])
        for x in ax:
            x.axvspan(a, b, color="#2f855a" if r["Spread %"] > 0 else "#c53030", alpha=.15, lw=0)
    for x in ax:
        x.grid(alpha=.25); x.set_xlim(idx[0], idx[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "relval_dax_spx.png"), dpi=130, facecolor="white")
    print("\nChart -> relval_dax_spx.png")


if __name__ == "__main__":
    main()
