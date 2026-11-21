"""
ZEW logic on Bunds, and DAX + Bund switching
============================================

Part A - Bund only.  Tests the MIRROR of the DAX rules on a 10-year Bund:
  ON  : ZEW crosses BELOW 0, or "renewed weakness" (ZEW turns down after an up
        month while the peak was still at a low level)
  OFF : ZEW above 0, or ZEW rises on 2 consecutive releases
and an alternative found in the event study:
  ON  : ZEW has fallen on 2 consecutive releases while still >= 0
plus candle filters on the Bund itself.

Part B - Switching.  DAX leg = zew_regime_strategy.BASE.  When the DAX leg is
in cash, hold Bunds subject to a rule, otherwise cash.

Bund = total-return proxy of a constant-maturity 10y Bund built from the daily
Bundesbank 10y zero-coupon yield (data/bund_10y_yield.csv, from Aug 1997).
Monthly returns correlate 0.97 with the iShares German government bond ETF.
Daily closes, next-close execution, 0.10% per switch, cash at the 3m rate.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zew_regime_strategy as zr
from zew_surprise_strategy import load_zew, load_dax, COST, HERE

START, SPLIT = "1997-09-01", "2012-01-01"


def load_bund(maturity=10.0) -> pd.Series:
    y = pd.read_csv(os.path.join(HERE, "data", "bund_10y_yield.csv"),
                    index_col=0, parse_dates=True)["yield"] / 100
    dt = y.index.to_series().diff().dt.days.values / 365.25
    n, ret = np.arange(1, int(maturity) + 1), [0.0]
    for i in range(1, len(y)):
        c, yn = y.iloc[i - 1], y.iloc[i]          # par bond bought yesterday
        t = n - dt[i]
        ret.append((c / (1 + yn) ** t).sum() + 1 / (1 + yn) ** t[-1] - 1)
    return (1 + pd.Series(ret, index=y.index)).cumprod().rename("bund")


def per_release_to_daily(e, flags):
    d, K = np.zeros(e.n, bool), len(flags)
    for k in range(K):
        d[e.rel_i[k]:(e.rel_i[k + 1] if k + 1 < K else e.n)] = flags[k]
    return d


def falling_regime(e, falls=2, above_zero=True):
    """ZEW has fallen `falls` releases in a row (optionally while still >= 0)."""
    zv, dz = e.zv, e.dz
    f = [k >= falls + 1 and all(dz[k - j] < 0 for j in range(falls))
         and (zv[k] >= 0 or not above_zero) for k in range(len(zv))]
    return per_release_to_daily(e, f)


def event_study(e, zew, px):
    prev, d1 = zew.shift(1), zew.diff()
    dn, pup = d1 < 0, d1.shift(1) > 0
    sets = {"All releases": pd.Series(True, index=zew.index),
            "Cross below 0": (zew < 0) & (prev >= 0),
            "Renewed weakness, peak <= 20": dn & pup & (prev <= 20),
            "Renewed weakness, peak <= 0": dn & pup & (prev <= 0),
            "ZEW < 0 and falling": (zew < 0) & dn,
            "ZEW < 0 and rising": (zew < 0) & ~dn,
            "ZEW >= 0 and falling": (zew >= 0) & dn,
            "ZEW >= 0 and rising": (zew >= 0) & ~dn}
    rows = []
    for name, m in sets.items():
        row = {"Set": name}
        for lab, h in [("3m", 63), ("6m", 126), ("12m", 252)]:
            r = np.array([px.iloc[i + h] / px.iloc[i] - 1
                          for i, ok, d in zip(e.rel_i, m.values, zew.index)
                          if ok and d >= px.index[0] and i + h < e.n])
            row["N"] = len(r)
            row[f"{lab} avg %"], row[f"{lab} hit %"] = r.mean() * 100, (r > 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("Set")


def main():
    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.2f}".format)
    zew = load_zew()
    bund, dax = load_bund(), load_dax()
    idx = bund.index.intersection(dax.index)
    bund, dax = bund.reindex(idx), dax.reindex(idx)
    LB, LD = zr.Lab(zew, bund), zr.Lab(zew, dax)
    LM = zr.Lab(-zew, bund)                        # mirrored ZEW -> mirrored rules
    e = LB.e
    bh_b = pd.Series(LB.e.ret, index=idx)[START:]
    bh_d = pd.Series(LD.e.ret, index=idx)[START:]

    def row(lab, r, pos=None):
        s = lab.stats(r)
        s["Sharpe h1"] = lab.stats(r[:SPLIT])["Sharpe"]
        s["Sharpe h2"] = lab.stats(r[SPLIT:])["Sharpe"]
        if pos is not None:
            s["Invested %"] = pos[idx >= START].mean() * 100
        return s

    def run(lab, dec):
        pos = np.roll(dec, 1).astype(float); pos[0] = 0
        return lab.returns(pos)[START:], pos

    print(f"Period {START} -> {idx[-1].date()}   (h1 = to {SPLIT}, h2 = after)")
    print("\n== A1. EVENT STUDY: Bund forward total return after ZEW releases ==")
    print(event_study(e, zew, bund).to_string())

    print("\n== A2. BUND ONLY ==")
    mirror = LM.regime("cross_or_surprise", 2, -20)   # peak <= 20
    fall2 = falling_regime(e)
    tests = {"Bund buy & hold": np.ones(e.n, bool),
             "Mirror (cross<0 / renewed weakness)": mirror,
             "Mirror + last month up": mirror & LB.mom["m_up1"],
             "Mirror + 2 of 3 months up": mirror & LB.mom["m_2of3"],
             "Mirror + stop 2 down / re-entry 1 up": mirror & LB.mom["m_SL2_RE1"],
             "Mirror + above 10-month average": mirror & LB.mom["m_>sma10"],
             "Mirror + above 40-week average": mirror & LB.mom["w_>sma40"],
             "Trend only (above 10-month average)": LB.mom["m_>sma10"],
             "ZEW falling 2x while >= 0": fall2,
             "ZEW falling 2x while >= 0 + stop/re-entry": fall2 & LB.mom["m_SL2_RE1"],
             "ZEW falling 2x while >= 0 + 2 of 3 up": fall2 & LB.mom["m_2of3"],
             "ZEW falling 2x (any level)": falling_regime(e, above_zero=False)}
    out = {}
    for k, dec in tests.items():
        r, pos = run(LB, dec)
        out[k] = row(LB, r, pos)
    print(pd.DataFrame(out).T.to_string())

    print("\n== B. DAX + BUND SWITCHING ==")
    dpos = LD.position(**zr.BASE)
    rd, rb, rf = LD.e.ret, LB.e.ret, LD.rf

    def switch(bdec):
        bpos = np.roll(bdec, 1).astype(float); bpos[0] = 0
        bpos *= 1 - dpos                                   # DAX has priority
        hd, hb = np.roll(dpos, 1), np.roll(bpos, 1); hd[0] = hb[0] = 0
        turn = np.abs(np.diff(dpos, prepend=0)) + np.abs(np.diff(bpos, prepend=0))
        r = hd * rd + hb * rb + (1 - hd - hb) * rf - turn * COST
        return pd.Series(r, index=idx)[START:], bpos

    out = {"DAX buy & hold": row(LD, bh_d), "Bund buy & hold": row(LD, bh_b),
           "60/40 DAX/Bund": row(LD, 0.6 * bh_d + 0.4 * bh_b),
           "DAX strategy, else cash": row(LD, LD.returns(dpos)[START:])}
    legs = {"else Bund always": np.ones(e.n, bool),
            "else Bund if above 10-month avg": LB.mom["m_>sma10"],
            "else Bund if mirror rule": mirror,
            "else Bund if mirror rule + trend": mirror & LB.mom["m_>sma10"],
            "else Bund if ZEW falling 2x >= 0": fall2,
            "else Bund if ZEW falling 2x + trend": falling_regime(e, above_zero=False) & LB.mom["m_>sma10"]}
    res = {}
    for k, dec in legs.items():
        r, bpos = switch(dec)
        res[k] = (r, bpos)
        out["DAX strategy, " + k] = {**row(LD, r), "Bund %": bpos[idx >= START].mean() * 100}
    tab = pd.DataFrame(out).T
    tab.to_csv(os.path.join(HERE, "zew_bund_switching.csv"))
    print(tab.to_string())

    pick = "else Bund if ZEW falling 2x >= 0"
    r, bpos = res[pick]
    yr = pd.DataFrame({"Switching %": r, "DAX strat %": LD.returns(dpos)[START:],
                       "DAX %": bh_d, "Bund %": bh_b}).groupby(lambda d: d.year).apply(
        lambda x: ((1 + x).prod() - 1) * 100)
    print(f"\nCalendar years - chosen: DAX strategy, {pick}")
    print(yr.T.to_string())
    state = "DAX" if dpos[-1] else "BUND" if bpos[-1] else "CASH"
    print(f"\nCurrent state ({idx[-1].date()}): {state}")

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3]})
    series = [("DAX buy & hold", bh_d, "#2b6cb0", 1.2), ("Bund buy & hold", bh_b, "#b7791f", 1.2),
              ("DAX strategy, else cash", LD.returns(dpos)[START:], "#68d391", 1.6),
              ("DAX strategy, else Bund while ZEW falling from above 0", r, "#22543d", 2.4)]
    for name, s, col, lw in series:
        c = (1 + s).cumprod()
        ax[0].plot(c.index, c, color=col, lw=lw, label=name)
        if lw != 1.6:
            ax[1].plot(c.index, (c / c.cummax() - 1) * 100, color=col, lw=1.2, label=name)
    m = idx >= START
    ax[0].fill_between(idx[m], 0, 1, where=dpos[m] > 0, color="#2b6cb0", alpha=.10, lw=0,
                       transform=ax[0].get_xaxis_transform(), label="In DAX")
    ax[0].fill_between(idx[m], 0, 1, where=bpos[m] > 0, color="#b7791f", alpha=.13, lw=0,
                       transform=ax[0].get_xaxis_transform(), label="In Bund")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)")
    ax[0].set_title("ZEW-driven DAX / Bund switching", fontweight="bold")
    ax[1].set_ylabel("Drawdown %")
    ax[0].legend(loc="upper left", fontsize=9, ncol=2)
    ax[1].legend(loc="lower right", fontsize=9, ncol=3)
    for a in ax:
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "zew_bund_backtest.png"), dpi=130, facecolor="white")
    print("Chart -> zew_bund_backtest.png")


if __name__ == "__main__":
    main()
