"""
Drawdown and recovery times for the Defensive + Growth combo (DAX + Bund).
An episode runs from a new high (peak) to the low (trough) and back to the
old high (recovery).  Times are in calendar months.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import best_combo_search as bc
from combo_portfolio import PICKS
from zew_surprise_strategy import COST, HERE

MIN_DD = 5.0          # list episodes deeper than this (%)


def episodes(r: pd.Series) -> pd.DataFrame:
    c = (1 + r).cumprod()
    peak = c.cummax()
    dd = c / peak - 1
    rows, i, n = [], 0, len(c)
    under = dd.values < 0
    while i < n:
        if not under[i]:
            i += 1
            continue
        j = i
        while j < n and under[j]:
            j += 1
        seg = dd.iloc[i:j]
        p, t = c.index[i - 1] if i else c.index[0], seg.idxmin()
        rec = c.index[j] if j < n else pd.NaT
        end = rec if j < n else c.index[-1]
        mo = lambda a, b: (b - a).days / 30.44
        rows.append({"Peak": p.date(), "Trough": t.date(), "Recovered": rec.date() if j < n else "not yet",
                     "Depth %": seg.min() * 100, "Fall (months)": mo(p, t),
                     "Recovery (months)": mo(t, end), "Total (months)": mo(p, end), "open": j >= n})
        i = j
    return pd.DataFrame(rows)


def summary(r: pd.Series) -> dict:
    e = episodes(r)
    big = e[e["Depth %"] <= -MIN_DD]
    c = (1 + r).cumprod()
    dd = c / c.cummax() - 1
    return {"Max drawdown %": dd.min() * 100,
            f"Episodes deeper than {MIN_DD:.0f}%": len(big),
            "Episodes deeper than 10%": int((e["Depth %"] <= -10).sum()),
            "Median total length (months)": big["Total (months)"].median(),
            "Average total length (months)": big["Total (months)"].mean(),
            "Longest total length (months)": big["Total (months)"].max(),
            "Longest recovery from trough (months)": big["Recovery (months)"].max(),
            "Time below previous high %": (dd < 0).mean() * 100,
            "Time more than 5% below high %": (dd < -0.05).mean() * 100,
            "Time more than 10% below high %": (dd < -0.10).mean() * 100}


def main():
    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.1f}".format)
    D = bc.build()
    idx, rd, rb, rf = D["idx"], D["rd"], D["rb"], D["rf"]
    R, w = {}, {}
    for name, (z, m, op, sc, f, g, b) in PICKS.items():
        R[name], *w[name] = bc.simulate(D, bc.dax_decision(D, z, m, op, sc, f, g), D["B"][b])
    wd, wb = (w["Defensive"][0] + w["Growth"][0]) / 2, (w["Defensive"][1] + w["Growth"][1]) / 2
    hd, hb = np.roll(wd, 1), np.roll(wb, 1); hd[0] = hb[0] = 0
    turn = np.abs(np.diff(wd, prepend=0)) + np.abs(np.diff(wb, prepend=0))
    R["Combo 50/50"] = hd * rd + hb * rb + (1 - hd - hb) * rf - turn * COST
    R["DAX buy & hold"], R["60/40 DAX/Bund"] = rd, 0.6 * rd + 0.4 * rb
    R = pd.DataFrame(R, index=idx)[["Combo 50/50", "Defensive", "Growth", "DAX buy & hold", "60/40 DAX/Bund"]]

    print(f"Period {idx[0].date()} -> {idx[-1].date()}")
    print("\n== SUMMARY =="); print(pd.DataFrame({k: summary(R[k]) for k in R}).to_string())
    e = episodes(R["Combo 50/50"])
    big = e[e["Depth %"] <= -MIN_DD].sort_values("Peak").drop(columns="open").reset_index(drop=True)
    print(f"\n== COMBO 50/50: every drawdown deeper than {MIN_DD:.0f}% =="); print(big.to_string())
    for k in ["Defensive", "Growth", "DAX buy & hold"]:
        t = episodes(R[k]); t = t[t["Depth %"] <= -10].drop(columns="open").reset_index(drop=True)
        print(f"\n== {k}: drawdowns deeper than 10% =="); print(t.to_string())

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1.5, 2]})
    c = (1 + R).cumprod()
    for k, col, lw in [("DAX buy & hold", "#a0aec0", 1.0), ("Combo 50/50", "#1a365d", 1.8)]:
        ax[0].fill_between(idx, (c[k] / c[k].cummax() - 1) * 100, 0, color=col, alpha=.55 if lw > 1 else .35, lw=0, label=k)
    ax[0].set_ylabel("Below previous high, %"); ax[0].set_xlim(idx[0], idx[-1])
    ax[0].set_title("Combo 50/50 (Defensive + Growth): drawdowns and recovery times", fontweight="bold")
    ax[0].legend(loc="lower left", fontsize=9); ax[0].grid(alpha=.25)
    b = e[e["Depth %"] <= -MIN_DD].sort_values("Peak").reset_index(drop=True)
    y = np.arange(len(b))[::-1]
    ax[1].barh(y, b["Fall (months)"], color="#c53030", label="Peak to trough")
    ax[1].barh(y, b["Recovery (months)"], left=b["Fall (months)"], color="#2f855a", label="Trough to new high")
    for yi, row in zip(y, b.itertuples()):
        ax[1].text(row._7 + 0.4, yi, f"{row._4:.1f}%   {row._7:.0f} mo" + ("  (still open)" if row.open else ""),
                   va="center", fontsize=9)
    ax[1].set_yticks(y); ax[1].set_yticklabels([f"{p:%b %Y}" for p in pd.to_datetime(b["Peak"])])
    ax[1].set_xlabel("Months"); ax[1].set_ylabel("Drawdown starting at"); ax[1].set_xlim(0, b["Total (months)"].max() * 1.25)
    ax[1].legend(loc="lower right", fontsize=9); ax[1].grid(alpha=.25, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "recovery_times.png"), dpi=130, facecolor="white")
    print("\nChart -> recovery_times.png")


if __name__ == "__main__":
    main()
