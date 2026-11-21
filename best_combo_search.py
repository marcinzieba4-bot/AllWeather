"""
Best-combination search: DAX + Bund, ZEW, MACD, candles/knots, SMA filters
==========================================================================

Every combination of the building blocks below is simulated on daily data.
To keep the search honest the winner is selected on the TRAIN period only
and then judged on the TEST period it never saw.

DAX LEG   long when  GATE  and  FILTER  (and optional GLOBAL SMA filter)
  ZEW gate  (checked on monthly ZEW releases)
      above_zero | cross (cross above 0) | surprise | cross_or_surprise
      (latched gates switch off below 0 or after 2 consecutive ZEW falls)
  MACD gate  monthly or weekly MACD(12,26) line rising
  gate logic ZEW only | MACD only | ZEW OR MACD | ZEW AND MACD | always
  filter     monthly / weekly candle rules, SMA rules, or the knot rules
             (previous-candle + wick entries, stop below low, re-entry)
             applied to the ZEW leg only or to the whole DAX leg
BUND LEG  held only when the DAX leg is out
      cash | always | ZEW falling 2x (>=0 / any) | mirror | Bund SMA / MACD

All decisions are taken at a close and executed at the NEXT close.
0.10% per switch, cash at the German 3m rate.
"""
import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import candle_macd_strategy as cm
import zew_bund_strategy as zb
import zew_regime_strategy as zr
from zew_surprise_strategy import load_zew, load_dax, COST, HERE

START, SPLIT = "1997-09-01", "2012-01-01"


def build():
    zew = load_zew()
    dax_full, bund = load_dax(), zb.load_bund()
    idx = bund.index.intersection(dax_full.index)
    idx = idx[idx >= START]
    LDf = zr.Lab(zew, dax_full)                      # filters on full DAX history
    LB = zr.Lab(zew, bund)
    LM = zr.Lab(-zew, bund)
    S = cm.Sim()

    def al(arr, src_idx):                            # align a daily array to idx
        return pd.Series(np.asarray(arr, float), index=src_idx).reindex(idx, method="ffill").fillna(0).values > 0

    def macd_rising(px, freq):
        per = px.index.to_period(freq)
        c = px.groupby(per).last()
        line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        up = line.diff() > 0
        end = np.append(per[1:] != per[:-1], True)
        out, cur = np.zeros(len(px), bool), False
        for i in range(len(px)):
            if end[i]:
                cur = bool(up.loc[per[i]])
            out[i] = cur
        return out

    Z = {k: al(LDf.regime(k), LDf.idx) for k in ["above_zero", "cross", "surprise", "cross_or_surprise"]}
    M = {"macd_M": al(macd_rising(dax_full, "M"), dax_full.index),
         "macd_W": al(macd_rising(dax_full, "W-FRI"), dax_full.index)}
    F = {k: al(v, LDf.idx) for k, v in LDf.mom.items()}
    for tf in ["W", "M"]:                            # knot rules: stop below low + re-entry
        F[f"knots_{tf}"] = al(S.go(tf=tf, regime="none_long", stop="low", reentry=True)[2], S.idx)
    G = {"none": np.ones(len(idx), bool), "m_>sma10": F["m_>sma10"], "w_>sma40": F["w_>sma40"]}
    eB = LB.e
    B = {"cash": np.zeros(LB.n, bool), "always": np.ones(LB.n, bool),
         "zew_fall2_pos": zb.falling_regime(eB), "zew_fall2_any": zb.falling_regime(eB, above_zero=False),
         "mirror": LM.regime("cross_or_surprise", 2, -20),
         "bund_sma10": LB.mom["m_>sma10"], "bund_macd_M": macd_rising(bund, "M"),
         "zew_fall2_any&sma10": zb.falling_regime(eB, above_zero=False) & LB.mom["m_>sma10"]}
    B = {k: al(v, LB.idx) for k, v in B.items()}
    rd = dax_full.pct_change().reindex(idx).fillna(0).values
    rb = bund.pct_change().reindex(idx).fillna(0).values
    rf = pd.Series(LDf.rf, index=LDf.idx).reindex(idx).fillna(0).values
    return dict(idx=idx, Z=Z, M=M, F=F, G=G, B=B, rd=rd, rb=rb, rf=rf)


def structures(Z, M):
    out = [("always", None, None, "all")]
    for z in Z:
        out.append((z, None, None, "all"))
        for m in M:
            out += [(z, m, "AND", "all"), (z, m, "OR", "zew_leg"), (z, m, "OR", "all")]
    out += [(None, m, None, "all") for m in M]
    return out


def dax_decision(D, z, m, op, scope, f, g):
    F, ones = D["F"][f], np.ones(len(D["idx"]), bool)
    if z == "always":
        dec = F
    elif m is None:
        dec = D["Z"][z] & F
    elif z is None:
        dec = D["M"][m] & F
    elif op == "AND":
        dec = D["Z"][z] & D["M"][m] & F
    elif scope == "zew_leg":
        dec = (D["Z"][z] & F) | D["M"][m]
    else:
        dec = (D["Z"][z] | D["M"][m]) & F
    return dec & D["G"][g]


def simulate(D, ddec, bdec):
    dpos = np.roll(ddec, 1).astype(float); dpos[0] = 0
    bpos = np.roll(bdec, 1).astype(float); bpos[0] = 0
    bpos *= 1 - dpos
    hd, hb = np.roll(dpos, 1), np.roll(bpos, 1); hd[0] = hb[0] = 0
    turn = np.abs(np.diff(dpos, prepend=0)) + np.abs(np.diff(bpos, prepend=0))
    return hd * D["rd"] + hb * D["rb"] + (1 - hd - hb) * D["rf"] - turn * COST, dpos, bpos


def stats(r, rf):
    c = np.cumprod(1 + r)
    sd = r.std()
    return ((c[-1] ** (252 / len(r)) - 1) * 100, (r - rf).mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
            (c / np.maximum.accumulate(c) - 1).min() * 100)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
    D = build()
    idx, rf = D["idx"], D["rf"]
    tr, te = idx < SPLIT, idx >= SPLIT
    t0, rows = time.time(), []
    for (z, m, op, scope), f, g, b in itertools.product(structures(D["Z"], D["M"]), D["F"], D["G"], D["B"]):
        r, dpos, bpos = simulate(D, dax_decision(D, z, m, op, scope, f, g), D["B"][b])
        a, s1, s2 = stats(r, rf), stats(r[tr], rf[tr]), stats(r[te], rf[te])
        rows.append(dict(zew=z or "-", macd=m or "-", logic=op or "-", scope=scope, filt=f, glob=g, bund=b,
                         cagr=a[0], sharpe=a[1], mdd=a[2], cagr_tr=s1[0], sh_tr=s1[1], mdd_tr=s1[2],
                         cagr_te=s2[0], sh_te=s2[1], mdd_te=s2[2],
                         dax_pct=dpos.mean() * 100, bund_pct=bpos.mean() * 100,
                         switches=int(np.abs(np.diff(dpos)).sum() + np.abs(np.diff(bpos)).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "best_combo_grid.csv"), index=False)
    print(f"{len(df)} combinations simulated in {time.time() - t0:.0f}s | {idx[0].date()} -> {idx[-1].date()}"
          f" | TRAIN to {SPLIT}, TEST after")

    bench = {"DAX buy & hold": D["rd"], "Bund buy & hold": D["rb"], "60/40 DAX/Bund": 0.6 * D["rd"] + 0.4 * D["rb"]}
    bt = pd.DataFrame({k: dict(zip(["CAGR", "Sharpe", "MaxDD", "CAGR train", "Sharpe train", "CAGR test", "Sharpe test", "MaxDD test"],
                                   stats(v, rf) + stats(v[tr], rf[tr])[:2] + stats(v[te], rf[te])))
                       for k, v in bench.items()}).T
    print("\n== BENCHMARKS =="); print(bt.to_string())
    b6040 = stats(bench["60/40 DAX/Bund"][te], rf[te])[1]

    keys = ["zew", "macd", "logic", "scope", "filt", "glob", "bund"]
    cols = keys + ["cagr", "sharpe", "mdd", "cagr_tr", "sh_tr", "cagr_te", "sh_te", "mdd_te", "dax_pct", "bund_pct", "switches"]
    print("\n== 1. DOES SELECTING ON TRAIN CARRY OVER TO TEST? ==")
    print(f"Rank correlation of train Sharpe vs test Sharpe across all combos: {df['sh_tr'].rank().corr(df['sh_te'].rank()):.2f}")
    top = df.sort_values("sh_tr", ascending=False)
    for n in [1, 10, 50, 200]:
        t = top.head(n)
        print(f"Top {n:3d} by TRAIN Sharpe -> median TEST Sharpe {t['sh_te'].median():.2f}, CAGR {t['cagr_te'].median():.1f}%, "
              f"MaxDD {t['mdd_te'].median():.0f}%   (all combos: {df['sh_te'].median():.2f}, 60/40: {b6040:.2f})")
    print("\nTop 10 by TRAIN Sharpe:"); print(top.head(10)[cols].to_string(index=False))

    print("\n== 2. WHICH BUILDING BLOCKS MATTER? median across all combos using each option ==")
    for k in keys:
        if k == "scope":
            continue
        print(df.groupby(k)[["sharpe", "sh_tr", "sh_te", "cagr", "mdd"]].median().sort_values("sharpe", ascending=False).to_string(), "\n")

    print("== 3. MOST ROBUST: best worst-half Sharpe (min of train, test) ==")
    df["sh_min"] = df[["sh_tr", "sh_te"]].min(axis=1)
    rob = df.sort_values("sh_min", ascending=False)
    print(rob.head(15)[cols].to_string(index=False))

    print("\n== 4. HIGHEST RETURN with MaxDD better than -25% and both halves Sharpe > 0.5 ==")
    q = df[(df["mdd"] > -25) & (df["sh_min"] > 0.5)].sort_values("cagr", ascending=False)
    print(q.head(10)[cols].to_string(index=False))

    print("\n== 5. HOW HARD IS THE TEST PERIOD? ==")
    bd = stats(bench["DAX buy & hold"][te], rf[te])
    print(f"Combos with TEST Sharpe above 60/40 ({b6040:.2f}): {(df['sh_te'] > b6040).sum()} of {len(df)}"
          f" | best TEST Sharpe of any combo: {df['sh_te'].max():.2f}")
    print(f"Combos with TEST CAGR above DAX buy & hold ({bd[0]:.1f}%): {(df['cagr_te'] > bd[0]).sum()} of {len(df)}")
    print(f"Combos with full-period Sharpe above 60/40 ({bt.loc['60/40 DAX/Bund', 'Sharpe']:.2f}): "
          f"{(df['sharpe'] > bt.loc['60/40 DAX/Bund', 'Sharpe']).sum()} of {len(df)}")

    picks = {"DEFENSIVE: ZEW + candle stop, Bund on falling ZEW":
                 ("cross_or_surprise", None, None, "all", "m_SL2_RE1", "none", "zew_fall2_pos"),
             "GROWTH: ZEW or monthly MACD, Bund on falling ZEW":
                 ("cross_or_surprise", "macd_M", "OR", "zew_leg", "m_SL2_RE1", "none", "zew_fall2_pos")}
    print("\n== 6. THE TWO RECOMMENDED COMBINATIONS ==")
    res, tab = {}, {}
    for name, (z, m, op, sc, f, g, b) in picks.items():
        r, dpos, bpos = simulate(D, dax_decision(D, z, m, op, sc, f, g), D["B"][b])
        res[name] = (r, dpos, bpos)
        a_, s1, s2 = stats(r, rf), stats(r[tr], rf[tr]), stats(r[te], rf[te])
        tab[name] = {"CAGR": a_[0], "Sharpe": a_[1], "MaxDD": a_[2], "CAGR train": s1[0], "Sharpe train": s1[1],
                     "CAGR test": s2[0], "Sharpe test": s2[1], "MaxDD test": s2[2],
                     "Test Sharpe percentile": (df["sh_te"] < s2[1]).mean() * 100,
                     "DAX %": dpos.mean() * 100, "Bund %": bpos.mean() * 100, "Cash %": (1 - dpos - bpos).mean() * 100}
    print(pd.DataFrame(tab).T.to_string())
    yrs = pd.DataFrame({**{k.split(":")[0].title() + " %": pd.Series(v[0], index=idx) for k, v in res.items()},
                        "DAX %": pd.Series(D["rd"], index=idx), "60/40 %": pd.Series(bench["60/40 DAX/Bund"], index=idx)}
                       ).groupby(lambda d: d.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print(yrs.T.to_string())
    print(f"Losing years: " + ", ".join(f"{c} {(yrs[c] < 0).sum()}" for c in yrs.columns)
          + " | worst year: " + ", ".join(f"{c} {yrs[c].min():.1f}" for c in yrs.columns))
    for name, (r, dpos, bpos) in res.items():
        print(f"Current state {idx[-1].date()} - {name.split(':')[0]}: "
              f"{'DAX' if dpos[-1] else 'BUND' if bpos[-1] else 'CASH'}")

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    lines = [("DAX buy & hold", D["rd"], "#2b6cb0", 1.2), ("60/40 DAX/Bund", bench["60/40 DAX/Bund"], "#718096", 1.2)]
    lines += [(k, v[0], c, 2.2) for (k, v), c in zip(res.items(), ["#22543d", "#c05621"])]
    for name, r, col, lw in lines:
        c = np.cumprod(1 + r)
        ax[0].plot(idx, c, color=col, lw=lw, label=name)
        ax[1].plot(idx, (c / np.maximum.accumulate(c) - 1) * 100, color=col, lw=1.1)
    ax[0].axvline(pd.Timestamp(SPLIT), color="black", ls="--", lw=1)
    ax[0].text(pd.Timestamp(SPLIT), 0.98, "  test period (not used for selection) →", transform=ax[0].get_xaxis_transform(),
               va="top", fontsize=9)
    ax[1].axvline(pd.Timestamp(SPLIT), color="black", ls="--", lw=1)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)"); ax[1].set_ylabel("Drawdown %")
    ax[0].set_title("Best DAX + Bund combinations from the full search", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    for a_ in ax:
        a_.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "best_combo_backtest.png"), dpi=130, facecolor="white")
    print("Chart -> best_combo_backtest.png")
    return D, df


if __name__ == "__main__":
    main()
