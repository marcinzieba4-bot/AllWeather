"""
Combined portfolio: 50% Defensive + 50% Growth  (DAX + Bund, ZEW / MACD rules)
==============================================================================
Defensive : DAX when ZEW regime is on and the monthly candle stop is not hit;
            Bund when out of DAX and ZEW has fallen twice while >= 0.
Growth    : same, plus DAX whenever the monthly MACD line is rising.
The combo holds the average of the two allocations (so DAX weight is 0, 50 or
100%), trading only the net change.  Next-close execution, 0.10% per unit of
turnover, cash at the German 3m rate.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import best_combo_search as bc
from zew_surprise_strategy import COST, HERE

PICKS = {"Defensive": ("cross_or_surprise", None, None, "all", "m_SL2_RE1", "none", "zew_fall2_pos"),
         "Growth": ("cross_or_surprise", "macd_M", "OR", "zew_leg", "m_SL2_RE1", "none", "zew_fall2_pos")}


def risk_stats(r: pd.Series, rf: pd.Series) -> dict:
    c = (1 + r).cumprod()
    dd = c / c.cummax() - 1
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    x = r - rf
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    y = (1 + r).groupby(r.index.year).prod() - 1
    under = (dd < 0).astype(int)
    longest = under.groupby((under == 0).cumsum()).sum().max()
    r12 = c.pct_change(252).dropna()
    return {"CAGR %": cagr * 100, "Volatility %": r.std() * np.sqrt(252) * 100,
            "Sharpe": x.mean() / r.std() * np.sqrt(252),
            "Sortino": x.mean() * 252 / (np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(252)),
            "Max drawdown %": dd.min() * 100, "Calmar (CAGR/MaxDD)": cagr / -dd.min(),
            "Longest drawdown (years)": longest / 252,
            "Worst day %": r.min() * 100, "Worst month %": m.min() * 100,
            "Monthly VaR 95% %": m.quantile(0.05) * 100,
            "Worst 12 months %": r12.min() * 100, "Best 12 months %": r12.max() * 100,
            "Positive months %": (m > 0).mean() * 100,
            "Worst year %": y.min() * 100, "Best year %": y.max() * 100,
            "Losing years": int((y < 0).sum()), "Growth of 1": c.iloc[-1]}


def main():
    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.2f}".format)
    D = bc.build()
    idx, rd, rb, rf = D["idx"], D["rd"], D["rb"], D["rf"]
    rets, w = {}, {}
    for name, (z, m, op, sc, f, g, b) in PICKS.items():
        r, dpos, bpos = bc.simulate(D, bc.dax_decision(D, z, m, op, sc, f, g), D["B"][b])
        rets[name], w[name] = r, (dpos, bpos)
    wd = (w["Defensive"][0] + w["Growth"][0]) / 2
    wb = (w["Defensive"][1] + w["Growth"][1]) / 2
    hd, hb = np.roll(wd, 1), np.roll(wb, 1); hd[0] = hb[0] = 0
    turn = np.abs(np.diff(wd, prepend=0)) + np.abs(np.diff(wb, prepend=0))
    rets["COMBO 50/50"] = hd * rd + hb * rb + (1 - hd - hb) * rf - turn * COST
    rets["DAX buy & hold"], rets["60/40 DAX/Bund"] = rd, 0.6 * rd + 0.4 * rb
    R = pd.DataFrame(rets, index=idx)[["COMBO 50/50", "Defensive", "Growth", "DAX buy & hold", "60/40 DAX/Bund"]]
    rfs = pd.Series(rf, index=idx)

    print(f"Period {idx[0].date()} -> {idx[-1].date()}")
    print("\n== RISK STATISTICS, full period ==")
    print(pd.DataFrame({k: risk_stats(R[k], rfs) for k in R}).to_string())
    for lab, sl in [("1997-2011", slice(None, "2011-12-31")), ("2012-2026", slice("2012-01-01", None))]:
        t = pd.DataFrame({k: risk_stats(R[k][sl], rfs[sl]) for k in R}).loc[
            ["CAGR %", "Volatility %", "Sharpe", "Max drawdown %", "Worst year %"]]
        print(f"\n== {lab} =="); print(t.to_string())

    print("\n== COMBO ALLOCATION ==")
    cash = 1 - wd - wb
    print(f"Average: DAX {wd.mean()*100:.0f}% | Bund {wb.mean()*100:.0f}% | cash {cash.mean()*100:.0f}%")
    print(f"Days at 100% DAX {(wd == 1).mean()*100:.0f}% | 50% DAX {(wd == .5).mean()*100:.0f}% | 0% DAX {(wd == 0).mean()*100:.0f}%")
    print(f"Turnover {turn.sum() / (len(idx) / 252):.1f}x per year | correlation Defensive vs Growth "
          f"{R['Defensive'].corr(R['Growth']):.2f} | combo vs DAX {R['COMBO 50/50'].corr(R['DAX buy & hold']):.2f}")
    print(f"Today ({idx[-1].date()}): DAX {wd[-1]*100:.0f}% | Bund {wb[-1]*100:.0f}% | cash {cash[-1]*100:.0f}%")

    yr = ((1 + R).groupby(R.index.year).prod() - 1) * 100
    print("\n== CALENDAR YEARS % =="); print(yr.T.to_string())

    # ── chart ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(4, 1, figsize=(15, 17), gridspec_kw={"height_ratios": [3.2, 1.3, 0.9, 1.6]})
    col = {"COMBO 50/50": ("#1a365d", 2.6), "Defensive": ("#2f855a", 1.2), "Growth": ("#c05621", 1.2),
           "DAX buy & hold": ("#a0aec0", 1.2), "60/40 DAX/Bund": ("#718096", 1.0)}
    C = (1 + R).cumprod()
    for k, (c_, lw) in col.items():
        ax[0].plot(C.index, C[k], color=c_, lw=lw, label=f"{k}  (x{C[k].iloc[-1]:.1f})",
                   ls="--" if k == "60/40 DAX/Bund" else "-", zorder=3 if k.startswith("COMBO") else 2)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log scale)")
    ax[0].set_title("Combined portfolio: 50% Defensive + 50% Growth — cumulative PnL", fontweight="bold", fontsize=13)
    ax[0].legend(loc="upper left", fontsize=10)
    for k in ["DAX buy & hold", "60/40 DAX/Bund", "COMBO 50/50"]:
        ax[1].plot(C.index, (C[k] / C[k].cummax() - 1) * 100, color=col[k][0], lw=1.6 if k.startswith("COMBO") else 1.0,
                   ls="--" if k == "60/40 DAX/Bund" else "-", label=k)
    ax[1].set_ylabel("Drawdown %"); ax[1].legend(loc="lower left", fontsize=9, ncol=3)
    ax[2].stackplot(idx, wd * 100, wb * 100, cash * 100, colors=["#2b6cb0", "#d69e2e", "#e2e8f0"],
                    labels=["DAX", "Bund", "Cash"], step="post")
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("Allocation %"); ax[2].legend(loc="upper left", fontsize=9, ncol=3)
    for a in ax[:3]:
        a.set_xlim(idx[0], idx[-1]); a.grid(alpha=.25)
    x = np.arange(len(yr))
    ax[3].bar(x - .2, yr["COMBO 50/50"], .4, color="#1a365d", label="COMBO 50/50")
    ax[3].bar(x + .2, yr["DAX buy & hold"], .4, color="#a0aec0", label="DAX buy & hold")
    ax[3].axhline(0, color="black", lw=.8); ax[3].set_xticks(x); ax[3].set_xticklabels(yr.index, rotation=45, fontsize=9)
    ax[3].set_ylabel("Calendar-year return %"); ax[3].legend(loc="upper right", fontsize=9); ax[3].grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "combo_portfolio.png"), dpi=130, facecolor="white")
    print("\nChart -> combo_portfolio.png")


if __name__ == "__main__":
    main()
