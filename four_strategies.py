"""
Four separate ZEW strategies, each on its own (no combining)
============================================================
1. DAX  - surprise at the upper band : ZEW turns UP after a down month, trough >= 30
2. DAX  - crossing zero from below   : ZEW crosses above 0
3. Bund - surprise (mirror)          : ZEW turns DOWN after an up month, peak <= 20
4. Bund - crossing zero from above   : ZEW crosses below 0
Exit (DAX) : ZEW below 0 or two consecutive falls.   Exit (Bund): mirrored.
Each is shown pure and with the monthly candle stop on the traded asset
(out after 2 down months, back in after 1 up month).

Daily closes, real ZEW release dates, next-close execution, 0.10% per switch.
% statistics include the cash rate while flat; EUR figures are futures-style
(no interest on capital) per EUR 100k of fixed stake.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zew_bund_strategy as zb
import zew_regime_strategy as zr
from combo_portfolio import risk_stats
from recovery_times import episodes, summary
from zew_surprise_strategy import load_zew, load_dax, HERE

START, STAKE = "1997-09-01", 100_000


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.2f}".format)
    zew, dax, bund = load_zew(), load_dax(), zb.load_bund()
    LD, LB, LM = zr.Lab(zew, dax), zr.Lab(zew, bund), zr.Lab(-zew, bund)
    defs = {"DAX | surprise at upper band": (LD, LD.regime("surprise", 2, 30)),
            "DAX | cross above zero": (LD, LD.regime("cross")),
            "Bund | surprise (renewed weakness)": (LB, LM.regime("surprise", 2, -20)),
            "Bund | cross below zero": (LB, LM.regime("cross"))}
    R, RF, POS, n_sig = {}, {}, {}, {}
    for name, (lab, reg) in defs.items():
        for tag, dec in [("pure", reg), ("with candle stop", reg & lab.mom["m_SL2_RE1"])]:
            pos = np.roll(dec, 1).astype(float); pos[0] = 0
            k = f"{name} | {tag}"
            R[k] = lab.returns(pos)[START:]
            RF[k] = pd.Series(lab.rf, index=lab.idx)[START:]
            POS[k] = pd.Series(pos, index=lab.idx)[START:]
        r_ = pd.Series(reg.astype(int), index=lab.idx)[START:]
        n_sig[name] = int((r_.diff() > 0).sum())
    R["DAX buy & hold"] = pd.Series(LD.e.ret, index=LD.idx)[START:]; RF["DAX buy & hold"] = pd.Series(LD.rf, index=LD.idx)[START:]
    R["Bund buy & hold"] = pd.Series(LB.e.ret, index=LB.idx)[START:]; RF["Bund buy & hold"] = pd.Series(LB.rf, index=LB.idx)[START:]

    print(f"Period {START} -> {LD.idx[-1].date()}")
    print("Regime switch-ons: " + ", ".join(f"{k}: {v}" for k, v in n_sig.items()))
    rows = {}
    for k, r in R.items():
        s = risk_stats(r, RF[k])
        ex = (r - RF[k]) * STAKE
        cum = ex.cumsum()
        rows[k] = {"CAGR %": s["CAGR %"], "Vol %": s["Volatility %"], "Sharpe": s["Sharpe"], "Sortino": s["Sortino"],
                   "MaxDD %": s["Max drawdown %"], "Worst month %": s["Worst month %"], "Worst year %": s["Worst year %"],
                   "Best year %": s["Best year %"], "Losing years": s["Losing years"], "Pos. months %": s["Positive months %"],
                   "Invested %": POS[k].mean() * 100 if k in POS else 100.0,
                   "EUR/yr per 100k": ex.sum() / (len(ex) / 252), "Max EUR DD per 100k": (cum - cum.cummax()).min()}
    print("\n== RISK STATISTICS =="); print(pd.DataFrame(rows).T.to_string())

    print("\n== RECOVERY TIMES (months; length stats cover drawdowns deeper than 5%) ==")
    print(pd.DataFrame({k: summary(r) for k, r in R.items()}).T.to_string(float_format="{:.1f}".format))
    for k in [x for x in R if x.endswith("with candle stop") and x.startswith("DAX")] + [x for x in R if x.endswith("pure") and x.startswith("Bund")]:
        e = episodes(R[k]).sort_values("Depth %").head(5).drop(columns="open").reset_index(drop=True)
        print(f"\n-- {k}: five deepest drawdowns --"); print(e.to_string(float_format="{:.1f}".format))

    yr = pd.DataFrame({k.replace(" | with candle stop", " +stop").replace(" | pure", ""): (1 + r).groupby(r.index.year).prod() - 1
                       for k, r in R.items() if "pure" in k or "buy" in k}) * 100
    print("\n== CALENDAR YEARS %, pure versions =="); print(yr.T.to_string(float_format="{:.1f}".format))

    fig, ax = plt.subplots(4, 2, figsize=(16, 17), gridspec_kw={"width_ratios": [2, 1.3]})
    for i, name in enumerate(defs):
        bh = "DAX buy & hold" if name.startswith("DAX") else "Bund buy & hold"
        for k, col, lw in [(bh, "#a0aec0", 1.1), (f"{name} | pure", "#c05621", 1.5), (f"{name} | with candle stop", "#1a365d", 2.0)]:
            c = (1 + R[k]).cumprod()
            lab_ = k.split(" | ")[-1] if k != bh else bh
            ax[i, 0].plot(c.index, c, color=col, lw=lw, label=f"{lab_}  (x{c.iloc[-1]:.1f})")
            ax[i, 1].plot(c.index, (c / c.cummax() - 1) * 100, color=col, lw=1.0 if k == bh else 1.3)
        p = POS[f"{name} | pure"]
        ax[i, 0].fill_between(p.index, 0, 1, where=p.values > 0, color="#c05621", alpha=.10, lw=0, transform=ax[i, 0].get_xaxis_transform())
        ax[i, 0].set_yscale("log"); ax[i, 0].set_title(name, fontweight="bold", loc="left"); ax[i, 0].legend(loc="upper left", fontsize=9)
        ax[i, 0].set_ylabel("Growth of 1 (log)"); ax[i, 1].set_ylabel("Drawdown %"); ax[i, 1].set_title("Drawdown", loc="left", fontsize=10)
        ax[i, 0].grid(alpha=.25); ax[i, 1].grid(alpha=.25)
    fig.suptitle("Four separate ZEW strategies (shaded = regime on)", fontweight="bold", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "four_strategies.png"), dpi=120, facecolor="white")
    print("\nChart -> four_strategies.png")


if __name__ == "__main__":
    main()
