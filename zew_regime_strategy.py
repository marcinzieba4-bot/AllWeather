"""
ZEW Regime + Candle Momentum DAX Strategy
=========================================

Two layers (same architecture as the ISM-MACD strategy):

LAYER 1 - ZEW regime (macro gate), evaluated on each ZEW release
  ON  : ZEW Economic Sentiment crosses ABOVE 0,  or
        "positive surprise": ZEW turns up after a down month with trough >= 30
  OFF : ZEW below 0,  or  ZEW falls on 2 consecutive releases

LAYER 2 - candle momentum (inside an ON regime), monthly DAX candles
  STOP     : 2 consecutive down months  -> cash
  RE-ENTRY : 1 up month                 -> long again (if regime still ON)

Mechanics: daily closes, real ZEW release dates, decisions executed at the
NEXT close, 0.10% cost per switch, cash earns the German 3-month rate.

Data: data/zew_konjunktur.xlsx, data/dax_daily.csv,
      data/de_3m_rate.csv (FRED IR3TIB01DEM156N)
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zew_surprise_strategy import Engine, load_zew, load_dax, COST, START, SPLIT, HERE

RATE_FILE = os.path.join(HERE, "data", "de_3m_rate.csv")
BASE = dict(regime="cross_or_surprise", falls=2, surprise_level=30, mom="m_SL2_RE1")
BORROW_SPREAD = 0.005      # financing spread over cash for the leveraged variants


class Lab:
    def __init__(self):
        self.e = e = Engine(load_zew(), load_dax())
        self.idx, self.n, self.px = e.idx, e.n, e.px
        rf = pd.read_csv(RATE_FILE, index_col=0, parse_dates=True).iloc[:, 0] / 100
        rf.index = rf.index + pd.offsets.MonthBegin(1)        # known one month later
        self.rf = rf.reindex(self.idx, method="ffill").fillna(0).values / 252
        self.mom = self._momentum_filters()

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    def regime(self, kind="cross_or_surprise", falls=2, surprise_level=30) -> np.ndarray:
        zv, dz, K = self.e.zv, self.e.dz, len(self.e.zv)
        r, on = np.zeros(K, bool), False
        for k in range(falls + 1, K):
            cross = zv[k] >= 0 > zv[k - 1]
            surprise = dz[k] > 0 and dz[k - 1] < 0 and zv[k - 1] >= surprise_level
            falling = all(dz[k - j] < 0 for j in range(falls))
            if kind == "always":
                on = True
            elif kind == "above_zero":
                on = zv[k] >= 0
            else:
                trig = cross if kind == "cross" else surprise if kind == "surprise" \
                    else (cross or surprise)
                if trig:
                    on = True
                elif on and (zv[k] < 0 or falling):
                    on = False
            r[k] = on
        daily = np.zeros(self.n, bool)
        ri = self.e.rel_i
        for k in range(K):
            daily[ri[k]:(ri[k + 1] if k + 1 < K else self.n)] = r[k]
        return daily

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    def _momentum_filters(self) -> dict:
        idx, px, n = self.idx, self.px, self.n

        def candles(freq):
            per = idx.to_period(freq)
            return per, px.groupby(per).last(), np.append(per[1:] != per[:-1], True)

        def asof(s, per, end):                 # value of last COMPLETED candle
            out, cur = np.zeros(n, bool), False
            for i in range(n):
                if end[i]:
                    cur = bool(s.get(per[i], False))
                out[i] = cur
            return out

        def stop_reentry(up, sl, re):
            u, s, on = up.values, pd.Series(False, index=up.index), True
            for i in range(max(sl, re), len(u)):
                if on and not u[i - sl + 1:i + 1].any():
                    on = False
                elif not on and u[i - re + 1:i + 1].all():
                    on = True
                s.iloc[i] = on
            return s

        pm, cm, em = candles("M")
        pw, cw, ew = candles("W-FRI")
        um, uw = cm.diff() > 0, cw.diff() > 0
        return {
            "none":        np.ones(n, bool),
            "m_up1":       asof(um, pm, em),                              # last month up
            "m_2of3":      asof(um.rolling(3).sum() >= 2, pm, em),
            "m_ret6>0":    asof(cm.pct_change(6) > 0, pm, em),
            "m_>sma10":    asof(cm > cm.rolling(10).mean(), pm, em),
            "m_SL2_RE1":   asof(stop_reentry(um, 2, 1), pm, em),
            "m_SL3_RE1":   asof(stop_reentry(um, 3, 1), pm, em),
            "w_3of5":      asof(uw.rolling(5).sum() >= 3, pw, ew),
            "w_>sma40":    asof(cw > cw.rolling(40).mean(), pw, ew),
            "w_SL3_RE2":   asof(stop_reentry(uw, 3, 2), pw, ew),
            "w_SL4_RE2":   asof(stop_reentry(uw, 4, 2), pw, ew),
        }

    # ── Backtest ─────────────────────────────────────────────────────────────
    def position(self, regime="cross_or_surprise", falls=2, surprise_level=30,
                 mom="m_SL2_RE1") -> np.ndarray:
        dec = self.regime(regime, falls, surprise_level) & self.mom[mom]
        pos = np.roll(dec, 1).astype(float)       # execute at next close
        pos[0] = 0
        return pos

    def returns(self, pos, lev=1.0) -> pd.Series:
        held = np.roll(pos, 1); held[0] = 0
        turn = np.abs(np.diff(pos, prepend=0))
        r = held * (lev * self.e.ret - (lev - 1) * (self.rf + BORROW_SPREAD / 252)) \
            + (1 - held) * self.rf - turn * COST * lev
        return pd.Series(r, index=self.idx)

    def stats(self, r: pd.Series) -> dict:
        rf = pd.Series(self.rf, index=self.idx).reindex(r.index)
        c = (1 + r).cumprod()
        x = r - rf
        return {"CAGR %": (c.iloc[-1] ** (252 / len(r)) - 1) * 100,
                "Vol %": r.std() * np.sqrt(252) * 100,
                "Sharpe": x.mean() / r.std() * np.sqrt(252),
                "MaxDD %": (c / c.cummax() - 1).min() * 100}


def trade_table(lab, pos):
    d = np.diff(pos, prepend=0)
    ins, outs = np.where(d > 0)[0], np.where(d < 0)[0]
    if len(outs) < len(ins):
        outs = np.append(outs, lab.n - 1)
    return pd.DataFrame([{"Entry": lab.idx[a].date(), "Exit": lab.idx[b].date(),
                          "Return %": (lab.px.iloc[b] / lab.px.iloc[a] - 1) * 100}
                         for a, b in zip(ins, outs)])


def plot(lab, pos, reg, r, bh, path):
    fig, ax = plt.subplots(3, 1, figsize=(15, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 2]})
    S, B = "#1f9d55", "#2b6cb0"
    cs, cb = (1 + r).cumprod(), (1 + bh).cumprod()
    ax[0].plot(cb.index, cb, color=B, lw=1.4, label="DAX buy & hold")
    ax[0].plot(cs.index, cs, color=S, lw=2.2, label="ZEW regime + candle momentum")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)")
    ax[0].set_title("ZEW regime (cross above 0 / positive surprise) + monthly candle stop — DAX",
                    fontweight="bold")
    ax[1].fill_between(cb.index, (cb / cb.cummax() - 1) * 100, 0, color=B, alpha=.3, label="DAX")
    ax[1].fill_between(cs.index, (cs / cs.cummax() - 1) * 100, 0, color=S, alpha=.6, label="Strategy")
    ax[1].set_ylabel("Drawdown %")
    z = lab.e.zew[START:]
    ax[2].plot(z.index, z, color="#dd6b20", lw=1.4, label="ZEW Economic Sentiment")
    ax[2].axhline(0, color="black", lw=.8)
    m = lab.idx >= START
    ax[2].fill_between(lab.idx[m], -70, 95, where=reg[m], color="#a0aec0", alpha=.25,
                       lw=0, label="ZEW regime ON")
    ax[2].set_ylabel("Balance"); ax[2].set_ylim(-70, 95)
    for a in ax[:2]:
        a.fill_between(lab.idx[m], 0, 1, where=pos[m] > 0, color=S, alpha=.10, lw=0,
                       transform=a.get_xaxis_transform(), label="Invested")
    for a in ax:
        a.grid(alpha=.25); a.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.2f}".format)
    lab = Lab()
    bh = pd.Series(lab.e.ret, index=lab.idx)[START:]
    pos = lab.position(**BASE)
    r = lab.returns(pos)[START:]

    print(f"== 1. BASE STRATEGY {BASE} ==  ({START} -> {lab.idx[-1].date()})")
    rows = {}
    for name, a, b in [("full", START, None), ("1992-2008", START, SPLIT), ("2009-2026", SPLIT, None)]:
        rows[f"Strategy {name}"] = lab.stats(r[a:b])
        rows[f"DAX B&H {name}"] = lab.stats(bh[a:b])
    print(pd.DataFrame(rows).T.to_string())
    inv = pos[lab.idx >= START].mean() * 100
    t = trade_table(lab, pos)
    t = t[pd.to_datetime(t["Entry"]) >= START]
    print(f"\nInvested {inv:.0f}% of days | {len(t)} trades | win {(t['Return %'] > 0).mean()*100:.0f}% | "
          f"avg {t['Return %'].mean():.1f}% | worst {t['Return %'].min():.1f}% | best {t['Return %'].max():.1f}%")

    print("\n== 2. WHAT EACH LAYER ADDS ==")
    layers = {"DAX buy & hold": bh,
              "ZEW above 0 only": lab.returns(lab.position("above_zero", mom="none"))[START:],
              "ZEW regime only": lab.returns(lab.position(mom="none"))[START:],
              "Candle momentum only": lab.returns(lab.position("always"))[START:],
              "ZEW regime + momentum": r,
              "  same, 1.5x leverage": lab.returns(pos, 1.5)[START:],
              "  same, 2.0x leverage": lab.returns(pos, 2.0)[START:]}
    print(pd.DataFrame({k: lab.stats(v) for k, v in layers.items()}).T.to_string())

    print("\n== 3. ROBUSTNESS: Sharpe by regime x momentum filter ==")
    regs = ["always", "above_zero", "cross", "surprise", "cross_or_surprise"]
    g = []
    for rg, mo in itertools.product(regs, lab.mom):
        rr = lab.returns(lab.position(rg, mom=mo))[START:]
        s = lab.stats(rr)
        g.append({"Regime": rg, "Momentum": mo, **s,
                  "Sharpe 92-08": lab.stats(rr[:SPLIT])["Sharpe"],
                  "Sharpe 09-26": lab.stats(rr[SPLIT:])["Sharpe"]})
    g = pd.DataFrame(g)
    g.to_csv(os.path.join(HERE, "zew_regime_grid.csv"), index=False)
    for col in ["Sharpe", "CAGR %", "MaxDD %"]:
        print(f"\n{col}")
        print(g.pivot(index="Momentum", columns="Regime", values=col)[regs].to_string())
    bs = lab.stats(bh)["Sharpe"]
    both = ((g["Sharpe 92-08"] > lab.stats(bh[:SPLIT])["Sharpe"]) &
            (g["Sharpe 09-26"] > lab.stats(bh[SPLIT:])["Sharpe"])).sum()
    print(f"\nVariants beating B&H Sharpe ({bs:.2f}) full sample: {(g['Sharpe'] > bs).sum()} of {len(g)}"
          f" | in BOTH halves: {both} of {len(g)}")

    print("\n== 4. NEIGHBOURHOOD of the base parameters ==")
    nb = []
    for f, L, mo in itertools.product([2, 3], [20, 30, 40], ["m_SL2_RE1", "m_SL3_RE1"]):
        rr = lab.returns(lab.position("cross_or_surprise", f, L, mo))[START:]
        nb.append({"ZEW falls": f, "Surprise lvl": L, "Momentum": mo, **lab.stats(rr),
                   "Sharpe 92-08": lab.stats(rr[:SPLIT])["Sharpe"],
                   "Sharpe 09-26": lab.stats(rr[SPLIT:])["Sharpe"]})
    print(pd.DataFrame(nb).to_string(index=False))

    print("\n== 5. CALENDAR YEARS ==")
    yr = pd.DataFrame({"Strategy %": r, "DAX %": bh}).groupby(lambda d: d.year).apply(
        lambda x: ((1 + x).prod() - 1) * 100)
    print(yr.T.to_string())
    print(f"Down years for DAX: {(yr['DAX %'] < 0).sum()} | strategy better in "
          f"{(yr['Strategy %'] > yr['DAX %'])[yr['DAX %'] < 0].sum()} of them")

    k = len(lab.e.zv) - 1
    print(f"\n== 6. CURRENT STATE ({lab.idx[-1].date()}) ==")
    print(f"ZEW {lab.e.zv[k]:.1f} (prev {lab.e.zv[k-1]:.1f}) | regime "
          f"{'ON' if lab.regime()[-1] else 'OFF'} | momentum "
          f"{'OK' if lab.mom[BASE['mom']][-1] else 'STOPPED'} | position "
          f"{'LONG DAX' if lab.regime()[-1] and lab.mom[BASE['mom']][-1] else 'CASH'}")

    plot(lab, pos, lab.regime(), r, bh, os.path.join(HERE, "zew_regime_backtest.png"))
    print("\nChart -> zew_regime_backtest.png")


if __name__ == "__main__":
    main()
