"""
Slow-MACD direction + candle / wick ("knot") entry strategy on DAX
==================================================================

REGIME   slow MACD line (EMA_fast - EMA_slow of candle closes).
         Rising  -> only longs.   Declining -> only shorts.

ENTRY    evaluated at the open of every candle, using the PREVIOUS candle:
  long A   previous candle closed up            -> long at the open
  long B   previous candle closed down, but price trades above
           PrevOpen + (PrevHigh - PrevOpen)/2   -> long there (stop-buy)
  shorts are the exact mirror.

STOP     "low"  : below the previous candle's low
         "half" : half-way into the previous candle's bottom wick
         "none" : no stop, position lasts until the setup disappears

RE-ENTRY after a stop-out, one re-entry per candle: a DAILY close back above
         the entry level (previous close for A, the wick level for B).
         New stop = lowest low of the current candle so far.

A position is carried into the next candle if the setup still holds there
(stop moves up to the new previous candle's low), otherwise it is closed at
the open.  Weekly / monthly candles are simulated on daily bars, so stops and
stop-buys fill intra-candle; gaps fill at the open.  If entry and stop are
both touched on the same day the trade is counted as stopped (conservative).

Costs 0.10% per trade side.  Flat earns the 3m rate, shorts earn -DAX + 2x
the 3m rate (short future on a total-return index plus collateral).
Data: data/dax_ohlc.csv (^GDAXI; real open/high/low from 1994).
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
START, SPLIT, COST = "1995-01-01", "2011-01-01", 0.001
BASE = dict(tf="M", regime="macd_long", macd=(12, 26), wick=False)   # what survived testing


class Sim:
    def __init__(self):
        d = pd.read_csv(os.path.join(HERE, "data", "dax_ohlc.csv"), index_col=0, parse_dates=True)
        self.d, self.idx, self.n = d, d.index, len(d)
        self.O, self.H, self.L, self.C = (d[c].values for c in ["Open", "High", "Low", "Close"])
        rf = pd.read_csv(os.path.join(HERE, "data", "de_3m_rate.csv"), index_col=0, parse_dates=True).iloc[:, 0] / 100
        rf.index = rf.index + pd.offsets.MonthBegin(1)
        self.rf = rf.reindex(self.idx, method="ffill").fillna(0).values / 252
        self.ret = pd.Series(self.C, index=self.idx).pct_change().fillna(0)
        self._zew = None

    def candles(self, tf):
        per = self.idx.to_period({"D": "D", "W": "W-FRI", "M": "M"}[tf])
        g = self.d.groupby(per)
        c = pd.DataFrame({"O": g["Open"].first(), "H": g["High"].max(),
                          "L": g["Low"].min(), "C": g["Close"].last()})
        cid = pd.Series(np.arange(len(c)), index=c.index).reindex(per).values
        return c, cid

    def zew_regime_by_candle(self, cid, ncand):
        """ZEW regime (cross above 0 / surprise, off after 2 falls) at each candle's last day."""
        if self._zew is None:
            import zew_regime_strategy as zr
            from zew_surprise_strategy import load_zew
            lab = zr.Lab(load_zew(), pd.Series(self.C, index=self.idx, name="dax"))
            self._zew = lab.regime()
        last = np.zeros(ncand, bool)
        last[cid] = self._zew          # last write per candle = its last day
        return last

    def run(self, tf="W", regime="macd_ls", macd=(12, 26), stop="low", reentry=True, wick=True):
        """regime: macd_ls | macd_long | none_long | zew_long | macd_zew_long.  wick=False -> hold regime."""
        c, cid = self.candles(tf)
        cO, cH, cL, cC = (c[k].values for k in "OHLC")
        line = (c["C"].ewm(span=macd[0], adjust=False).mean() - c["C"].ewm(span=macd[1], adjust=False).mean()).values
        mdir = np.sign(np.diff(line, prepend=np.nan))
        zreg = self.zew_regime_by_candle(cid, len(c)) if "zew" in regime else None
        O, H, L, C, n = self.O, self.H, self.L, self.C, self.n
        out = np.zeros(n)
        pos, ref, trades = 0, 0.0, []
        g = A = stopped = re_used = False
        trig = stp = Rlvl = ext = 0.0
        entry_px = 0.0

        def close(px):
            nonlocal pos
            f = 1 + pos * (px / ref - 1) - COST
            trades.append(pos * (px / entry_px - 1))
            pos = 0
            return f

        for i in range(1, n):
            f, t = 1.0, cid[i]
            prev_pos = pos
            if t != cid[i - 1] and t >= 2:                       # ── new candle opens
                p = t - 1
                if regime == "macd_ls":
                    g = int(mdir[p]) if not np.isnan(mdir[p]) else 0
                elif regime == "macd_long":
                    g = 1 if mdir[p] > 0 else 0
                elif regime == "none_long":
                    g = 1
                elif regime == "zew_long":
                    g = 1 if zreg[p] else 0
                else:                                             # macd_zew_long
                    g = 1 if (mdir[p] > 0 and zreg[p]) else 0
                stopped = re_used = False
                if g:
                    up = cC[p] > cO[p]
                    if g > 0:
                        A = up or not wick
                        trig = cO[p] + (cH[p] - cO[p]) / 2
                        stp = {"low": cL[p], "half": cL[p] + (min(cO[p], cC[p]) - cL[p]) / 2, "none": -np.inf}[stop]
                    else:
                        A = (cC[p] < cO[p]) or not wick
                        trig = cO[p] - (cO[p] - cL[p]) / 2
                        stp = {"low": cH[p], "half": cH[p] - (cH[p] - max(cO[p], cC[p])) / 2, "none": np.inf}[stop]
                    if not wick:
                        stp = -np.inf * g
                    Rlvl = cC[p] if A else trig
                    ext = O[i]
                want = bool(g) and (A or g * (O[i] - trig) > 0) and g * (O[i] - stp) > 0
                if pos and (pos != g or not want):
                    f *= close(O[i])
                if bool(g) and (A or g * (O[i] - trig) > 0) and not want:
                    stopped = True                                # gapped through the stop
                if not pos and want:
                    pos, ref, entry_px = g, O[i], O[i]
                    f *= 1 - COST
            if g:                                                 # ── intraday
                lo, hi = (L[i], H[i]) if g > 0 else (H[i], L[i])  # adverse / favourable extreme
                ext = min(ext, lo) if g > 0 else max(ext, lo)
                if pos:
                    if g * (lo - stp) < 0:
                        f *= close(O[i] if g * (O[i] - stp) < 0 else stp)
                        stopped = True
                elif not stopped and not A and g * (hi - trig) > 0:
                    px = O[i] if g * (O[i] - trig) > 0 else trig
                    pos, ref, entry_px = g, px, px
                    f *= 1 - COST
                    if g * (lo - stp) < 0:                        # same-day stop (conservative)
                        f *= close(stp)
                        stopped = True
                if stopped and reentry and not re_used and not pos and g * (C[i] - Rlvl) > 0:
                    pos, ref, entry_px, stp = g, C[i], C[i], ext
                    f *= 1 - COST
                    re_used, stopped = True, False
            if pos:
                f *= 1 + pos * (C[i] / ref - 1)
                ref = C[i]
            f += self.rf[i] * (1 if prev_pos == 0 else 2 if prev_pos < 0 else 0)
            out[i] = f - 1
            self._pos_log[i] = pos
        return pd.Series(out, index=self.idx), np.array(trades)

    _pos_log = None

    def go(self, **kw):
        self._pos_log = np.zeros(self.n)
        r, tr = self.run(**kw)
        return r, tr, self._pos_log.copy()

    def stats(self, r):
        rf = pd.Series(self.rf, index=self.idx).reindex(r.index)
        c = (1 + r).cumprod()
        return {"CAGR %": (c.iloc[-1] ** (252 / len(r)) - 1) * 100, "Vol %": r.std() * np.sqrt(252) * 100,
                "Sharpe": (r - rf).mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
                "MaxDD %": (c / c.cummax() - 1).min() * 100}

    def row(self, **kw):
        r, tr, pos = self.go(**kw)
        r, m = r[START:], self.idx >= START
        s = self.stats(r)
        s.update({"Sharpe h1": self.stats(r[:SPLIT])["Sharpe"], "Sharpe h2": self.stats(r[SPLIT:])["Sharpe"],
                  "Long %": (pos[m] > 0).mean() * 100, "Short %": (pos[m] < 0).mean() * 100,
                  "Trades": len(tr), "Win %": (tr > 0).mean() * 100 if len(tr) else np.nan,
                  "Avg trade %": tr.mean() * 100 if len(tr) else np.nan})
        return s


def main():
    pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", 300)
    S = Sim()
    bh = S.ret[START:]
    b = S.stats(bh)
    print(f"Period {START} -> {S.idx[-1].date()}  (h1 to {SPLIT}, h2 after)")
    print("DAX buy & hold:", {k: round(v, 2) for k, v in b.items()},
          "| Sharpe h1 %.2f h2 %.2f" % (S.stats(bh[:SPLIT])["Sharpe"], S.stats(bh[SPLIT:])["Sharpe"]))

    print("\n== 1. BUILDING BLOCKS (stop below low, re-entry on) ==")
    blocks = {}
    for tf in ["W", "M"]:
        blocks[f"{tf} | MACD direction only, long/short"] = S.row(tf=tf, regime="macd_ls", wick=False)
        blocks[f"{tf} | MACD direction only, long/flat"] = S.row(tf=tf, regime="macd_long", wick=False)
        blocks[f"{tf} | candle+wick logic only, long"] = S.row(tf=tf, regime="none_long")
        blocks[f"{tf} | MACD + candle+wick, long/short"] = S.row(tf=tf, regime="macd_ls")
        blocks[f"{tf} | MACD + candle+wick, long only"] = S.row(tf=tf, regime="macd_long")
        blocks[f"{tf} | ZEW regime + candle+wick, long"] = S.row(tf=tf, regime="zew_long")
        blocks[f"{tf} | MACD + ZEW + candle+wick, long"] = S.row(tf=tf, regime="macd_zew_long")
    print(pd.DataFrame(blocks).T.to_string())

    print("\n== 2. STOP AND RE-ENTRY VARIANTS (MACD + candle+wick, long only) ==")
    v = {}
    for tf, stop, re in itertools.product(["W", "M"], ["none", "low", "half"], [False, True]):
        if stop == "none" and re:
            continue
        v[f"{tf} | stop={stop:4s} | re-entry={'on' if re else 'off'}"] = S.row(tf=tf, regime="macd_long", stop=stop, reentry=re)
    print(pd.DataFrame(v).T.to_string())

    print("\n== 3. MACD SPEED (long only, stop low, re-entry on) ==")
    v = {}
    for tf, mc in itertools.product(["W", "M"], [(6, 13), (12, 26), (18, 39), (24, 52)]):
        v[f"{tf} | MACD{mc}"] = S.row(tf=tf, regime="macd_long", macd=mc)
    print(pd.DataFrame(v).T.to_string())

    print("\n== 4. DAILY CANDLES (for reference) ==")
    v = {"D | MACD(12,26) + candle+wick, long/short": S.row(tf="D", regime="macd_ls"),
         "D | MACD(12,26) + candle+wick, long only": S.row(tf="D", regime="macd_long"),
         "D | MACD(50,100) + candle+wick, long only": S.row(tf="D", regime="macd_long", macd=(50, 100))}
    print(pd.DataFrame(v).T.to_string())

    print("\n== 5. COST SENSITIVITY: same tests at 0.02% per side (futures-like) ==")
    global COST
    COST, v = 0.0002, {}
    for tf in ["D", "W", "M"]:
        v[f"{tf} | MACD direction only, long/flat"] = S.row(tf=tf, regime="macd_long", wick=False)
        v[f"{tf} | candle+wick logic only, long"] = S.row(tf=tf, regime="none_long")
        v[f"{tf} | MACD + candle+wick, long only"] = S.row(tf=tf, regime="macd_long")
        v[f"{tf} | MACD + candle+wick, long/short"] = S.row(tf=tf, regime="macd_ls")
    COST = 0.001
    print(pd.DataFrame(v).T[["CAGR %", "Sharpe", "MaxDD %", "Sharpe h1", "Sharpe h2", "Trades"]].to_string())

    print("\n== 6. MONTHLY MACD DIRECTION combined with the ZEW strategy ==")
    import zew_regime_strategy as zr
    from zew_surprise_strategy import load_zew
    lab = zr.Lab(load_zew(), pd.Series(S.C, index=S.idx, name="dax"))
    zpos = lab.position(**zr.BASE)
    mpos = S.go(**BASE)[2]
    combos = {"DAX buy & hold": np.ones(S.n), "ZEW strategy": zpos,
              "Monthly MACD rising (long/flat)": mpos,
              "ZEW AND MACD": zpos * mpos, "ZEW OR MACD": np.maximum(zpos, mpos),
              "50% ZEW + 50% MACD": 0.5 * zpos + 0.5 * mpos}
    out, rets = {}, {}
    for k, p_ in combos.items():
        rr = lab.returns(p_)[START:]
        rets[k] = rr
        out[k] = {**lab.stats(rr), "Sharpe h1": lab.stats(rr[:SPLIT])["Sharpe"],
                  "Sharpe h2": lab.stats(rr[SPLIT:])["Sharpe"], "Invested %": p_[S.idx >= START].mean() * 100}
    print(pd.DataFrame(out).T.to_string())
    r, pos = rets["ZEW OR MACD"], np.maximum(zpos, mpos)
    yr = pd.DataFrame({"ZEW OR MACD %": r, "DAX %": bh}).groupby(lambda d: d.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print(yr.T.to_string())
    print(f"Current ({S.idx[-1].date()}): ZEW leg {'LONG' if zpos[-1] else 'flat'}, "
          f"MACD leg {'LONG' if mpos[-1] else 'flat'}")

    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    lines = [("DAX buy & hold", bh, "#2b6cb0", 1.3),
             ("ZEW strategy OR monthly MACD rising", r, "#22543d", 2.3),
             ("Monthly MACD rising only (long/flat)", rets["Monthly MACD rising (long/flat)"], "#b7791f", 1.3),
             ("Monthly MACD + candle/wick entries + stop", S.go(**{**BASE, "wick": True})[0][START:], "#c53030", 1.3)]
    for name, s_, col, lw in lines:
        cu = (1 + s_).cumprod()
        ax[0].plot(cu.index, cu, color=col, lw=lw, label=name)
        ax[1].plot(cu.index, (cu / cu.cummax() - 1) * 100, color=col, lw=1.1)
    m = S.idx >= START
    ax[0].fill_between(S.idx[m], 0, 1, where=pos[m] > 0, color="#22543d", alpha=.08, lw=0,
                       transform=ax[0].get_xaxis_transform(), label="Invested (ZEW OR MACD)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)"); ax[1].set_ylabel("Drawdown %")
    ax[0].set_title("Slow MACD direction + previous-candle / wick entries — DAX", fontweight="bold")
    ax[0].legend(loc="upper left", fontsize=9)
    for a in ax:
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "candle_macd_backtest.png"), dpi=130, facecolor="white")
    print("Chart -> candle_macd_backtest.png")


if __name__ == "__main__":
    main()
