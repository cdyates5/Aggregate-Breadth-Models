#!/usr/bin/env python3
"""
rebuild.py - S&P 500 Breadth Aggregators (Macro Ops-style Breadth Components + Breadth Thrust Aggregator)

Regenerates the single-file dashboard from an empty directory:
    python rebuild.py                        # -> docs/index.html (served by GitHub Pages)
    python rebuild.py --out my.html          # anywhere else
    python rebuild.py --refresh              # ignore the price cache and re-download full history

Pipeline
  1. Point-in-time S&P 500 membership: fja05680/sp500 history file, rolled forward with Wikipedia's current table.
  2. Daily prices for every member since 1998 from Yahoo Finance (yfinance). Cached in ./cache; later runs
     download only the last three weeks and splice them on, re-basing adjusted history for dividends/splits.
  3. Bottom-up breadth: advances/declines, % above 10/20/50/200DMA, 20-day highs, dollar up-volume share,
     ratio-adjusted McClellan Oscillator and Summation Index.
  4. Component legs, thrust signals, aggregators and forward-return evidence.
  5. Data injected into the embedded HTML template (re.subn with a lambda replacement).

Requires: see requirements.txt   (pip install -r requirements.txt)
Known source quirks are listed in the dashboard's "Method and caveats" section.
"""
import argparse, io, json, os, pickle, re, sys, time, warnings
import datetime as dt
import numpy as np, pandas as pd, requests

warnings.filterwarnings("ignore")

# =============================================================== 1. membership + prices
CACHE = os.environ.get("BREADTH_CACHE", "cache")
START = "1998-01-01"
UA = {"User-Agent": "Mozilla/5.0 (research; breadth-dashboard)"}
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def get_membership():
    """Point-in-time membership: fja05680 history (1996->) rolled forward with Wikipedia changes."""
    txt = requests.get(MEMBERS_URL, headers=UA, timeout=60).text
    hist = pd.read_csv(io.StringIO(txt), parse_dates=["date"])
    hist["set"] = hist["tickers"].str.split(",").apply(lambda x: frozenset(t.strip() for t in x if t.strip()))
    hist = hist[["date", "set"]].sort_values("date").reset_index(drop=True)
    last_date, cur = hist["date"].iloc[-1], set(hist["set"].iloc[-1])
    notes = {"hist_last": str(last_date.date()), "wiki_applied": [], "wiki_ok": False}
    try:
        html = requests.get(WIKI_URL, headers=UA, timeout=60).text
        tab = pd.read_html(io.StringIO(html))[0]
        tab["sym"] = tab["Symbol"].astype(str).str.strip()
        tab["d"] = pd.to_datetime(tab["Date added"], errors="coerce")
        current = set(tab["sym"])
        adds = tab[tab["sym"].isin(current - cur)].copy()
        adds["d"] = adds["d"].fillna(last_date + pd.Timedelta(days=1)).clip(lower=last_date + pd.Timedelta(days=1))
        gone = sorted(cur - current)          # removal dates are not published in the table:
        rows = []                             # retire one leaver per joiner, in date order
        for d, g in adds.sort_values("d").groupby("d"):
            for sym in g["sym"]:
                cur.add(sym)
                out = gone.pop(0) if gone else ""
                cur.discard(out)
                notes["wiki_applied"].append(f"{d.date()} +{sym} -{out}")
            rows.append({"date": d, "set": frozenset(cur)})
        for out in gone:
            cur.discard(out)
        today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
        rows.append({"date": today, "set": frozenset(current)})
        hist = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
        notes["wiki_ok"] = True
    except Exception as e:  # history file alone is still usable
        notes["wiki_err"] = repr(e)
    return hist, notes


def ysym(t):
    return t.replace(".", "-")


def _dl(syms, **kw):
    import yfinance as yf
    for attempt in range(4):
        try:
            df = yf.download(syms, auto_adjust=False, progress=False, threads=True, group_by="column", timeout=30, **kw)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log("download retry", attempt + 1, repr(e)[:90])
        time.sleep(10 * (attempt + 1))
    return None


def _fetch(tickers, start):
    cl, ac, vo = {}, {}, {}
    tickers = sorted(tickers)
    B = 60
    for i in range(0, len(tickers), B):
        batch = tickers[i:i + B]
        df = _dl([ysym(t) for t in batch], start=start)
        if df is None:
            continue
        for t in batch:
            y = ysym(t)
            try:
                c = df["Close"][y].dropna() if isinstance(df.columns, pd.MultiIndex) else df["Close"].dropna()
                if len(c) == 0:
                    continue
                cl[t] = c
                ac[t] = (df["Adj Close"][y] if isinstance(df.columns, pd.MultiIndex) else df["Adj Close"]).dropna()
                vo[t] = (df["Volume"][y] if isinstance(df.columns, pd.MultiIndex) else df["Volume"]).dropna()
            except Exception:
                pass
        log(f"prices {min(i + B, len(tickers))}/{len(tickers)} ok={len(cl)} (from {start})")
        time.sleep(0.5)
    return cl, ac, vo


def _splice(old, new):
    """Join a fresh tail onto cached history. Yahoo re-bases adjusted prices after every dividend/split,
    so the cached history is rescaled to the fresh download on the first overlapping date."""
    if old is None or len(old) == 0:
        return new, 1.0
    if new is None or len(new) == 0:
        return old, 1.0
    common = old.index.intersection(new.index)
    ratio = 1.0
    if len(common):
        d0 = common[0]
        if old.loc[d0] and np.isfinite(old.loc[d0]) and np.isfinite(new.loc[d0]):
            ratio = float(new.loc[d0] / old.loc[d0])
        head = old[old.index < d0] * ratio
    else:
        head = old[old.index < new.index[0]]
    return pd.concat([head, new]), ratio


def get_prices(tickers, must=None):
    """Full download on a cold cache; otherwise only the last ~3 weeks for known tickers
    plus full history for tickers not seen before."""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, "prices.pkl")
    old = None
    if os.path.exists(fp):
        try:
            old = pickle.load(open(fp, "rb"))
        except Exception as e:
            log("cache unreadable, full download:", repr(e)[:80])
    tickers = set(tickers)
    if old is not None:
        last = old["adj"].index.max()
        if (pd.Timestamp.now() - last).days > 120:
            log("cache older than 120 days, full download"); old = None
    if old is None:
        cl, ac, vo = _fetch(tickers, START)
        tried = set(tickers)
        close, adj, vol = pd.DataFrame(cl), pd.DataFrame(ac), pd.DataFrame(vo)
    else:
        tried = set(old.get("tried", old["adj"].columns))
        known = sorted(tickers & set(old["adj"].columns))
        # active names only: delisted tickers return nothing and are kept as cached
        recent = old["adj"].index.max() - pd.Timedelta(days=30)
        live = [t for t in known if old["adj"][t].last_valid_index() is not None and old["adj"][t].last_valid_index() >= recent]
        start = (old["adj"].index.max() - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
        log(f"incremental update: {len(live)} active tickers from {start}; {len(tickers - tried)} new tickers in full")
        cl, ac, vo = _fetch(live, start)
        ncl, nac, nvo = _fetch(tickers - tried, START) if tickers - tried else ({}, {}, {})
        tried |= tickers
        close, adj, vol = old["close"].copy(), old["adj"].copy(), old["vol"].copy()
        cols_c, cols_a, cols_v = {}, {}, {}
        rescaled = 0
        for t in set(close.columns) | set(cl):
            oc = close[t].dropna() if t in close else None
            oa = adj[t].dropna() if t in adj else None
            ov = vol[t].dropna() if t in vol else None
            if t in cl:
                c, rc = _splice(oc, cl[t]); a_, ra = _splice(oa, ac[t])
                v, _ = _splice(ov / rc if ov is not None and rc else ov, vo[t])  # volume moves inversely to split ratio
                rescaled += abs(ra - 1) > 1e-6
                cols_c[t], cols_a[t], cols_v[t] = c, a_, v
            else:
                cols_c[t], cols_a[t], cols_v[t] = oc, oa, ov
        log(f"spliced {len(cl)} tickers; {rescaled} re-based for dividends/splits")
        close, adj, vol = pd.DataFrame(cols_c), pd.DataFrame(cols_a), pd.DataFrame(cols_v)
        if ncl:
            close = pd.concat([close, pd.DataFrame(ncl)], axis=1)
            adj = pd.concat([adj, pd.DataFrame(nac)], axis=1)
            vol = pd.concat([vol, pd.DataFrame(nvo)], axis=1)
    # second chance for current members still missing
    for t in sorted(set(must or ()) - set(adj.columns[adj.iloc[-10:].notna().any()])):
        df = _dl(ysym(t), start=START)
        if df is None:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) >= 30:
            close[t] = df["Close"]; adj[t] = df["Adj Close"]; vol[t] = df["Volume"]
            log("recovered", t)
    spx = _dl("^GSPC", start="1996-01-01")
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)
    out = {"close": close.sort_index(), "adj": adj.sort_index(), "vol": vol.sort_index(), "spx": spx, "tried": tried}
    pickle.dump(out, open(fp, "wb"))
    return out


# =============================================================== 2. breadth series
def build_breadth(px, hist, start="1998-01-01"):
    adj, close, vol, spx = px["adj"], px["close"], px["vol"], px["spx"]
    idx = spx.index[spx.index >= start]
    adj = adj.reindex(idx); close = close.reindex(idx); vol = vol.reindex(idx)
    # drop glitchy prints
    adj = adj.where(adj > 0)
    # point-in-time membership mask
    cols = adj.columns
    h = hist.sort_values("date")
    mrows = np.zeros((len(h), len(cols)), dtype=bool)
    cpos = {c: i for i, c in enumerate(cols)}
    for r, s in enumerate(h["set"]):
        for t in s:
            j = cpos.get(t)
            if j is not None:
                mrows[r, j] = True
    pos = np.searchsorted(h["date"].values, idx.values, side="right") - 1
    pos = np.clip(pos, 0, len(h) - 1)
    member = pd.DataFrame(mrows[pos], index=idx, columns=cols)
    n_listed = pd.Series([len(s) for s in h["set"]], index=h["date"].values)
    n_index = pd.Series(n_listed.values[pos], index=idx)

    ret = adj.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.9)  # bad-tick guard
    valid = member & ret.notna()
    adv = (valid & (ret > 0)).sum(axis=1)
    dec = (valid & (ret < 0)).sum(axis=1)
    n = valid.sum(axis=1)

    out = pd.DataFrame(index=idx)
    out["spx"] = spx["Close"].reindex(idx)
    out["n"] = n
    out["coverage"] = n / n_index
    out["adv"] = adv; out["dec"] = dec
    out["adv_share"] = adv / (adv + dec)

    dvol = (close * vol)
    upv = dvol.where(valid & (ret > 0)).sum(axis=1)
    dnv = dvol.where(valid & (ret < 0)).sum(axis=1)
    out["upvol_share"] = upv / (upv + dnv)

    for w in (10, 20, 50, 200):
        ma = adj.rolling(w, min_periods=w).mean()
        ok = member & ma.notna() & adj.notna()
        out[f"p{w}"] = 100 * (ok & (adj > ma)).sum(axis=1) / ok.sum(axis=1)
    hi20 = adj.rolling(20, min_periods=20).max()
    ok = member & hi20.notna() & adj.notna()
    out["h20"] = 100 * (ok & (adj >= hi20)).sum(axis=1) / ok.sum(axis=1)
    hi252 = adj.rolling(252, min_periods=200).max(); lo252 = adj.rolling(252, min_periods=200).min()
    ok = member & hi252.notna() & adj.notna()
    out["nh"] = 100 * (ok & (adj >= hi252)).sum(axis=1) / ok.sum(axis=1)
    out["nl"] = 100 * (ok & (adj <= lo252)).sum(axis=1) / ok.sum(axis=1)

    # McClellan, ratio-adjusted; Summation via the Miekka closed form (level is start-date independent)
    rana = 1000 * (adv - dec) / (adv + dec)
    e19 = rana.ewm(span=19, adjust=False).mean()
    e39 = rana.ewm(span=39, adjust=False).mean()
    out["osc"] = e19 - e39
    out["msi"] = 19 * e39 - 9 * e19
    return out.dropna(subset=["spx"]), member


# =============================================================== 3. signals, aggregators, evidence
EVID_START = "2000-01-03"
HZ = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
SPLIT = "2013-01-01"


def cross_up(s, lvl):
    return ((s > lvl) & (s.shift(1) <= lvl)).fillna(False)


def cross_dn(s, lvl):
    return ((s < lvl) & (s.shift(1) >= lvl)).fillna(False)


def cycled(s, lo, hi, win):
    return ((s > hi) & (s.shift(1) <= hi) & (s.rolling(win).min() < lo)).fillna(False)


def hold(trig, n):
    """Flag stays on for n sessions from the most recent trigger day."""
    return trig.astype(float).rolling(n, min_periods=1).max().fillna(0).astype(int)


def first_of_cluster(sig, gap=21):
    d = np.flatnonzero(sig.values)
    keep, last = [], -10 ** 9
    for i in d:
        if i - last > gap:
            keep.append(i)
        last = i
    return np.array(keep, dtype=int)


def build_model(b_full, notes, n_tickers):
    b = b_full[b_full.index >= "1999-01-04"].copy()
    spx = b["spx"]
    ev0 = b.index.get_indexer([pd.Timestamp(EVID_START)], method="bfill")[0]
    n = len(b)
    fwd = pd.DataFrame({k: spx.shift(-h) / spx - 1 for k, h in HZ.items()})
    sv = spx.values
    dd6 = np.array([sv[i:i + 127].min() / sv[i] - 1 if i + 126 < n else np.nan for i in range(n)])
    in_ev = np.arange(n) >= ev0

    def stats_events(idx):
        idx = idx[idx >= ev0]
        r = {"n": int(len(idx))}
        r["episodes"] = int(1 + (np.diff(idx) > 126).sum()) if len(idx) else 0
        for k in ("3m", "6m", "12m"):
            v = fwd[k].values[idx]; v = v[~np.isnan(v)]
            r[k + "_avg"] = round(float(v.mean() * 100), 1) if len(v) else None
            r[k + "_win"] = round(float((v > 0).mean() * 100)) if len(v) else None
        d = dd6[idx]; d = d[~np.isnan(d)]
        r["dd_worst"] = round(float(d.min() * 100), 1) if len(d) else None
        r["dd_med"] = round(float(np.median(d) * 100), 1) if len(d) else None
        dates = b.index[idx]
        for lab, m in (("h1", dates < SPLIT), ("h2", dates >= SPLIT)):
            v = fwd["12m"].values[idx[m]]; v = v[~np.isnan(v)]
            r["n_" + lab] = int(m.sum())
            r["12m_" + lab] = round(float(v.mean() * 100), 1) if len(v) else None
        # placebo: every circular shift of the whole event set (exact, no RNG)
        arr = fwd["12m"].values[ev0:]; ok = ~np.isnan(arr); m_ = int(ok.sum())
        ii = idx - ev0; ii = ii[ii < m_]
        if len(ii):
            obs = arr[ii].mean()
            sims = arr[(ii[None, :] + np.arange(m_)[:, None]) % m_].mean(axis=1)
            r["p12"] = round(float((sims >= obs - 1e-12).mean()), 3)
        else:
            r["p12"] = None
        r["last"] = str(dates[-1].date()) if len(dates) else None
        return r

    def stats_state(mask, cond=None):
        m = mask & in_ev if cond is None else mask & in_ev & cond
        den = in_ev.sum() if cond is None else (in_ev & cond).sum()
        r = {"pct": round(float(m.sum() / den * 100), 1) if den else None, "days": int(m.sum())}
        for k in ("1m", "3m", "6m", "12m"):
            v = fwd[k].values[m]; v = v[~np.isnan(v)]
            r[k] = round(float(v.mean() * 100), 1) if len(v) else None
        for k in ("6m", "12m"):
            v = fwd[k].values[m]; v = v[~np.isnan(v)]
            r["w" + k] = round(float((v > 0).mean() * 100)) if len(v) else None
        return r

    h1 = np.asarray(b.index < SPLIT); h2 = ~h1
    base = {"all": stats_state(np.ones(n, bool)), "h1": stats_state(np.ones(n, bool), h1), "h2": stats_state(np.ones(n, bool), h2)}

    # ---------------- thrust candidates ----------------
    ema10 = b["adv_share"].ewm(span=10, adjust=False).mean()
    a10 = b["adv"].rolling(10).sum() / b["dec"].rolling(10).sum()
    a20 = b["adv"].rolling(20).sum() / b["dec"].rolling(20).sum()
    adt = b["adv"].rolling(5).sum() / (b["adv"] + b["dec"]).rolling(5).sum()
    udt = b["upvol_share"].rolling(5).mean()
    up90 = ((b["upvol_share"] >= 0.9) & (b["adv_share"] >= 0.9))
    up80 = ((b["upvol_share"] >= 0.8) & (b["adv_share"] >= 0.8))
    dn90 = ((b["upvol_share"] <= 0.1) & (b["adv_share"] <= 0.1))
    C = [  # key, name, rule, family, trigger
        ("zbt", "Zweig Breadth Thrust", "10d EMA of adv/(adv+dec) from <0.40 to >0.615 inside 10 sessions", "A/D velocity", cycled(ema10, 0.40, 0.615, 10)),
        ("bam", "Deemer Breakaway Momentum", "10d advances ÷ 10d declines crosses above 1.97", "A/D magnitude", cross_up(a10, 1.97)),
        ("adt", "Whaley A/D Thrust", "5d advances ÷ (adv+dec) crosses above 73.66%", "A/D magnitude", cross_up(adt, 0.7366)),
        ("dg", "deGraaf 20-day-high thrust", "≥55% of members at a 20-day closing high", "New-high expansion", cross_up(b["h20"], 55)),
        ("p50", "%>50DMA above 90%", "share of members above 50DMA crosses above 90%", "Trend participation", cross_up(b["p50"], 90)),
        ("osc", "McClellan Oscillator thrust", "ratio-adjusted oscillator crosses above +90", "Breadth momentum", cross_up(b["osc"], 90)),
        ("bam20", "Deemer BAM (20d variant)", "20d advances ÷ 20d declines crosses above 1.72", "A/D magnitude", cross_up(a20, 1.72)),
        ("udt", "Whaley Up-Volume Thrust", "5d up-volume share crosses above 77.88% (dollar volume)", "Volume", cross_up(udt, 0.7788)),
        ("up90x2", "Two 90% up days in 10 sessions", "≥90% up-volume and ≥90% advancers, twice in 10d", "Volume", (up90 & (up90.rolling(10).sum() >= 2)).fillna(False)),
        ("up90", "Single 90% up day", "≥90% up-volume and ≥90% advancers", "Volume", up90.fillna(False)),
        ("up80b2b", "Back-to-back 80% up days", "Desmond/Lowry alternate reversal rule", "Volume", (up80 & up80.shift(1, fill_value=False)).fillna(False)),
        ("p20", "%>20DMA above 90%", "share of members above 20DMA crosses above 90%", "Trend participation", cross_up(b["p20"], 90)),
        ("p10", "%>10DMA <20% → >90%", "cycles from below 20% to above 90% inside 21 sessions", "Trend participation", cycled(b["p10"], 20, 90, 21)),
        ("msi500", "Summation −500 → +500", "ratio-adjusted Summation Index cycles inside 63 sessions", "Breadth momentum", cycled(b["msi"], -500, 500, 63)),
        ("msi1000", "Summation <0 → >+1000", "“escape velocity” style surge inside 63 sessions", "Breadth momentum", cycled(b["msi"], 0, 1000, 63)),
        ("p200", "%>200DMA <35% → >60%", "long-term participation rebuilt inside 126 sessions", "Trend participation", cycled(b["p200"], 35, 60, 126)),
    ]
    SELECTED = ["zbt", "adt", "dg", "p50", "osc"]
    cand = []
    trig = {}
    b12 = base["all"]["12m"]
    for key, name, rule, fam, t in C:
        trig[key] = t
        st = stats_events(first_of_cluster(t))
        passes = {
            "win": (st["12m_win"] or 0) >= 90,
            "p": st["p12"] is not None and st["p12"] < 0.05,
            "halves": all(st[k] is not None and st[k] >= b12 + 3 for k in ("12m_h1", "12m_h2")),
            "n": st["n"] >= 15,
        }
        st.update({"key": key, "name": name, "rule": rule, "family": fam, "pass": passes, "selected": key in SELECTED})
        cand.append(st)

    CAP = [
        ("Whaley A/D capitulation", "5d advances ÷ (adv+dec) below 19.05%", cross_dn(adt, 0.1905)),
        ("McClellan Oscillator below −100", "ratio-adjusted", cross_dn(b["osc"], -100)),
        ("%>50DMA below 10%", "", cross_dn(b["p50"], 10)),
        ("%>200DMA below 20%", "", cross_dn(b["p200"], 20)),
        ("Two 90% down days in 10 sessions", "≥90% down-volume and ≥90% decliners", (dn90 & (dn90.rolling(10).sum() >= 2)).fillna(False)),
    ]
    caps = []
    for name, rule, t in CAP:
        st = stats_events(first_of_cluster(t)); st.update({"name": name, "rule": rule}); caps.append(st)

    # ---------------- thrust aggregator ----------------
    HOLD_T = 63
    act = {k: hold(trig[k], HOLD_T) for k in SELECTED}
    count = sum(act.values())
    mx = count.rolling(21, min_periods=1).max()
    tstate = pd.Series(np.where(mx >= 2, 1, np.where(mx == 0, -1, 0)), index=b.index)
    tst = {}
    for v, lab in ((1, "bull"), (0, "neutral"), (-1, "bear")):
        m = (tstate == v).values
        tst[lab] = {"all": stats_state(m), "h1": stats_state(m, h1), "h2": stats_state(m, h2)}
    bycount = [dict(stats_state((count == c).values), count=c) for c in range(0, 6)]
    dret = spx.pct_change().shift(-1).values
    for v, lab in ((1, "bull"), (0, "neutral"), (-1, "bear")):
        m = (tstate == v).values & in_ev & ~np.isnan(dret)
        tst[lab]["ann"] = round(float(dret[m].mean() * 252 * 100), 1)
        tst[lab]["vol"] = round(float(dret[m].std() * np.sqrt(252) * 100), 1)

    # event log (selected signals, de-duplicated)
    log = []
    for key, name, *_ in C:
        if key not in SELECTED:
            continue
        for i in first_of_cluster(trig[key]):
            if i < ev0:
                continue
            log.append({"d": str(b.index[i].date()), "k": key, "s": name, "px": round(float(sv[i]), 0),
                        "f3": None if np.isnan(fwd["3m"].values[i]) else round(float(fwd["3m"].values[i] * 100), 1),
                        "f6": None if np.isnan(fwd["6m"].values[i]) else round(float(fwd["6m"].values[i] * 100), 1),
                        "f12": None if np.isnan(fwd["12m"].values[i]) else round(float(fwd["12m"].values[i] * 100), 1)})
    log.sort(key=lambda r: r["d"], reverse=True)

    # ---------------- breadth components ----------------
    hi252 = spx.rolling(252, min_periods=126).max()
    near = (spx >= 0.97 * hi252).fillna(False)
    p200mx = b["p200"].rolling(252, min_periods=126).max()
    tA = near & (b["p200"] <= p200mx - 20)
    tB = near & (b["msi"] < 250)
    tC = near & (b["p50"] < 50)
    t1 = ((b["osc"] <= -80) | (b["p50"] <= 20)).fillna(False)
    t2 = ((b["p50"] <= 10) | (b["p200"] <= 20)).fillna(False)
    HOLD_D, HOLD_W1, HOLD_W2 = 63, 21, 63
    dA, dB, dC = hold(tA, HOLD_D), hold(tB, HOLD_D), hold(tC, HOLD_D)
    w1, w2 = hold(t1, HOLD_W1), hold(t2, HOLD_W2)
    score = (1 - dA) + (1 - dB) + (1 - dC) + w1 + 2 * w2
    net = score - 3
    nearv = near.values
    legs = []
    for key, name, rule, t, a in (
        ("A", "Divergence A · %>200DMA", "S&P within 3% of its 52-week high while %>200DMA sits ≥20 pts below its own 52-week peak", tA, dA),
        ("B", "Divergence B · Summation Index", "S&P within 3% of its 52-week high while the ratio-adjusted Summation Index is below +250", tB, dB),
        ("C", "Divergence C · %>50DMA", "S&P within 3% of its 52-week high while fewer than half of members are above their 50DMA", tC, dC),
    ):
        legs.append({"key": key, "name": name, "rule": rule, "kind": "div",
                     "trig": {"all": stats_state(t.values), "h1": stats_state(t.values, h1), "h2": stats_state(t.values, h2)},
                     "active": stats_state(a.values.astype(bool))})
    near_base = {"all": stats_state(nearv), "h1": stats_state(nearv, h1), "h2": stats_state(nearv, h2)}
    for key, name, rule, t, a in (
        ("W1", "Oversold washout (+1)", "McClellan Oscillator ≤ −80 or %>50DMA ≤ 20%; stays on 21 sessions", t1, w1),
        ("W2", "Deep oversold washout (+2)", "%>50DMA ≤ 10% or %>200DMA ≤ 20%; stays on 63 sessions", t2, w2),
    ):
        legs.append({"key": key, "name": name, "rule": rule, "kind": "wash",
                     "trig": {"all": stats_state(t.values), "h1": stats_state(t.values, h1), "h2": stats_state(t.values, h2)},
                     "active": stats_state(a.values.astype(bool))})
    raw = [
        ("McClellan Oscillator ≤ −80", (b["osc"] <= -80)), ("%>50DMA ≤ 20%", (b["p50"] <= 20)), ("%>50DMA ≤ 10%", (b["p50"] <= 10)),
        ("%>200DMA ≤ 20%", (b["p200"] <= 20)), ("Summation Index ≤ −500", (b["msi"] <= -500)), ("Summation Index ≤ −1000", (b["msi"] <= -1000)),
    ]
    rawlv = [dict(stats_state(m.fillna(False).values), name=nm) for nm, m in raw]
    buckets = []
    for lab, m in (("0 · all three divergences, no washout", score == 0), ("1", score == 1), ("2", score == 2), ("3 · clean tape, no washout", score == 3),
                   ("4+ · washout", score >= 4)):
        mv = m.values
        buckets.append({"label": lab, "all": stats_state(mv), "h1": stats_state(mv, h1), "h2": stats_state(mv, h2)})

    # ---------------- payload ----------------
    sl = slice(ev0, n)
    idx = b.index[sl]

    def arr(s, nd=1):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else (int(v) if nd == 0 else round(float(v), nd)) for v in np.asarray(s)[sl]]

    def since(series):
        v = series.values; i = len(v) - 1
        while i > 0 and v[i - 1] == v[-1]:
            i -= 1
        return str(series.index[i].date())

    last = b.iloc[-1]
    P = {
        "meta": {
            "asof": str(b.index[-1].date()), "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "evid_start": str(idx[0].date()), "n_days": int(len(idx)), "split": SPLIT[:4],
            "members_last": int(last["n"]), "tickers_priced": int(n_tickers), "membership": notes,
            "cov_first": round(float(b["coverage"].iloc[ev0:ev0 + 252].mean() * 100)), "cov_2010": round(float(b.loc["2010", "coverage"].mean() * 100)),
            "cov_2020": round(float(b.loc["2020", "coverage"].mean() * 100)), "cov_last": round(float(b["coverage"].iloc[-21:].mean() * 100)),
            "hold_thrust": HOLD_T, "hold_div": HOLD_D, "hold_w1": HOLD_W1, "hold_w2": HOLD_W2,
        },
        "d": [str(x.date()) for x in idx],
        "spx": arr(spx, 2), "p50": arr(b["p50"]), "p200": arr(b["p200"]), "osc": arr(b["osc"]), "msi": arr(b["msi"], 0),
        "cov": arr(b["coverage"] * 100, 0),
        "dA": arr(dA, 0), "dB": arr(dB, 0), "dC": arr(dC, 0), "w1": arr(w1, 0), "w2": arr(w2, 0), "score": arr(score, 0),
        "tcount": arr(count, 0), "tstate": arr(tstate, 0), "act": {k: arr(act[k], 0) for k in SELECTED},
        "now": {
            "spx": round(float(last["spx"]), 2), "p50": round(float(last["p50"]), 1), "p200": round(float(last["p200"]), 1),
            "osc": round(float(last["osc"]), 1), "msi": round(float(last["msi"]), 0), "p200max": round(float(p200mx.iloc[-1]), 1),
            "dd": round(float((spx.iloc[-1] / hi252.iloc[-1] - 1) * 100), 1),
            "score": int(score.iloc[-1]), "net": int(net.iloc[-1]), "net_since": since(net),
            "dA": int(dA.iloc[-1]), "dB": int(dB.iloc[-1]), "dC": int(dC.iloc[-1]), "w1": int(w1.iloc[-1]), "w2": int(w2.iloc[-1]),
            "tcount": int(count.iloc[-1]), "tstate": int(tstate.iloc[-1]), "tstate_since": since(tstate), "tcount_since": since(count),
            "last_thrust": log[0] if log else None,
            "adt": round(float(adt.iloc[-1] * 100), 1), "a10": round(float(a10.iloc[-1]), 2), "ema10": round(float(ema10.iloc[-1]), 3), "h20": round(float(last["h20"]), 1),
            "last_trig": {k: (str(b.index[np.flatnonzero(t.values)[-1]].date()) if t.values.any() else None)
                          for k, t in (("A", tA), ("B", tB), ("C", tC), ("W1", t1), ("W2", t2))},
        },
        "base": base, "near_base": near_base, "cand": cand, "caps": caps, "tst": tst, "bycount": bycount, "log": log[:40],
        "legs": legs, "rawlv": rawlv, "buckets": buckets,
        "sel": [{"key": k, "name": next(c[1] for c in C if c[0] == k)} for k in SELECTED],
    }
    return P


# =============================================================== 4. dashboard template
TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>S&amp;P 500 Breadth Aggregators</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- House font is Bahnschrift (Light), a Windows system font; Barlow is the web fallback for machines without it. -->
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&display=swap" rel="stylesheet">
<meta name="color-scheme" content="light">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --paper:#FFFFFF; --card:#FFFFFF; --ink:#000000; --body:#262626; --muted:#808080; --hair:#D6E3F0;
  --navy:#1F4E79; --yellow:#FFC000; --hl:#FFFF00; --grid:#BDD7EE; --pale:#DEEBF7;
  --grey:#A5A5A5; --green:#70AD47; --blue:#4472C4; --accent:#ED7D31;
  --font:Bahnschrift,"Barlow","Segoe UI",Arial,sans-serif;
  color-scheme:light;
}
*{box-sizing:border-box}
html{background:#FFFFFF;}
body{margin:0;background:#FFFFFF;color:var(--body);font-family:var(--font);font-weight:300;line-height:1.45;-webkit-font-smoothing:antialiased;}
b,strong{font-weight:500;}
.wrap{max-width:1220px;margin:0 auto;padding:36px 26px 0;}
header{border-bottom:2px solid var(--navy);padding-bottom:18px;margin-bottom:26px;}
.kicker{font-size:13px;font-weight:400;letter-spacing:.06em;text-transform:uppercase;color:var(--navy);margin:0 0 14px;}
h1{font-weight:300;font-size:clamp(28px,4.2vw,44px);line-height:1.18;margin:0 0 14px;color:var(--ink);letter-spacing:0;}
h1 .hl{background:var(--hl);padding:0 .14em;-webkit-box-decoration-break:clone;box-decoration-break:clone;}
.sub{font-size:16px;color:#404040;max-width:80ch;margin:0 0 16px;}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:12.5px;color:var(--muted);}
.meta b{color:var(--ink);font-weight:400;}
/* Panels: navy header band, message in black, chart title grey + underlined + centred, legend under the chart */
.card{background:#FFFFFF;border:1px solid var(--hair);border-radius:0;padding:0 20px 16px;margin-bottom:22px;min-width:0;}
.eyebrow{background:var(--navy);color:#FFFFFF;font-size:12.5px;font-weight:400;letter-spacing:.02em;margin:0 -20px 14px;padding:7px 20px;display:flex;align-items:center;gap:10px;}
.livebadge{margin-left:auto;font-size:10.5px;font-weight:500;letter-spacing:.05em;background:var(--yellow);color:var(--navy);padding:1px 7px;border-radius:2px;white-space:nowrap;}
.livebadge::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--navy);margin-right:5px;vertical-align:middle;}
.card h3{font-weight:300;font-size:18px;color:var(--muted);text-align:center;text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:4px;margin:2px 0 10px;letter-spacing:0;}
.deck{font-size:15px;color:var(--ink);margin:0 0 12px;line-height:1.4;}
.deck p{margin:0 0 8px;} .deck p:last-child{margin:0;}
.deck b{font-weight:500;}
.chartbox{position:relative;width:100%;}
.legendrow{display:flex;flex-wrap:wrap;justify-content:center;gap:4px 16px;font-size:12.5px;color:var(--muted);margin:8px 0 4px;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.sq{border-radius:0;}
.cap{font-size:11.5px;color:var(--muted);margin:12px 0 0;padding-top:8px;border-top:1px solid #EDEDED;line-height:1.5;}
.cap b{color:var(--navy);font-weight:400;}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;}
.cols .card{margin-bottom:22px;}
@media(max-width:900px){.cols{grid-template-columns:1fr;gap:0;}}
/* current-read tiles */
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:2px 0 14px;}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr));}}
.kpi{border:1px solid #D9D9D9;border-radius:0;padding:10px 12px;background:#FFFFFF;}
.kpi.now{border:2px solid var(--navy);padding:9px 11px;}
.kpi .lab{font-size:11.5px;font-weight:400;letter-spacing:.06em;text-transform:uppercase;color:var(--navy);}
.kpi .val{font-size:30px;font-weight:300;color:var(--ink);line-height:1.2;margin-top:3px;font-variant-numeric:tabular-nums;}
.kpi .val small{font-size:14px;color:var(--muted);}
.kpi .note{font-size:12.5px;color:#404040;margin-top:5px;line-height:1.45;}
.pill{display:inline-block;font-size:10.5px;letter-spacing:.02em;padding:2px 7px;border-radius:2px;white-space:nowrap;font-weight:400;vertical-align:middle;}
.pill.number{background:var(--navy);color:#FFFFFF;}
.pill.neu{background:var(--pale);color:var(--navy);}
.pill.none{background:transparent;color:var(--muted);border:1px dashed #BFBFBF;}
.pill.bad{background:#C00000;color:#FFFFFF;}
.pill.good{background:#548235;color:#FFFFFF;}
/* window selector */
.bar{display:flex;align-items:center;gap:6px;margin:0 0 14px;position:sticky;top:0;background:#FFFFFF;padding:8px 0;z-index:5;}
.bar span{font-size:12.5px;color:var(--navy);text-transform:uppercase;letter-spacing:.06em;font-weight:400;margin-right:6px;}
.bar button{font-family:var(--font);font-weight:400;font-size:12px;padding:3px 12px;border:1px solid var(--grid);background:#FFFFFF;color:var(--navy);border-radius:2px;cursor:pointer;}
.bar button.on{background:var(--navy);color:#FFFFFF;border-color:var(--navy);}
/* tables */
.tblwrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
th{background:var(--navy);color:#FFFFFF;font-size:11.5px;font-weight:400;letter-spacing:.01em;text-align:right;padding:6px 8px;border:0;vertical-align:bottom;}
td{padding:7px 8px;border-bottom:1px solid var(--pale);vertical-align:top;line-height:1.35;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--body);}
th:first-child,td:first-child{text-align:left;}
td.t{text-align:left;white-space:normal;font-size:13.5px;font-weight:400;color:var(--ink);}
td.t small{display:block;color:var(--muted);font-size:11.5px;font-weight:300;margin-top:2px;line-height:1.35;}
tr.grp td{background:var(--pale);color:var(--navy);font-weight:400;font-size:12.5px;text-align:left;}
tr.base td, tr.base td.t{color:var(--muted);font-weight:300;}
tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--navy);}
.notes{font-size:13.5px;color:#404040;margin:0;padding-left:20px;}
.notes li{margin-bottom:7px;line-height:1.5;}
.notes b{color:var(--ink);}
.footbar{background:var(--navy);color:#FFFFFF;margin-top:26px;font-size:14px;font-weight:300;}
.footbar .in{max-width:1220px;margin:0 auto;padding:10px 26px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;}
@media(max-width:520px){.wrap{padding:28px 16px 0;}.footbar .in{padding:10px 16px;font-size:12.5px;}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="kicker">Systematic Macro · Market Breadth</p>
  <h1><span class="hl">S&amp;P 500 Breadth Aggregators</span></h1>
  <p class="sub">Macro Ops-style Breadth Components and Breadth Thrust Aggregator, rebuilt bottom-up from point-in-time index constituents. Components use the McClellan Summation Index, % of members above their 200- and 50-day averages and the McClellan Oscillator; the thrust count uses the five published thrust signals that survived forward-return testing.</p>
  <div class="meta" id="meta"></div>
</header>

<section class="card">
  <p class="eyebrow">Current read<span class="livebadge">LIVE · YAHOO</span></p>
  <div class="kpis" id="kpis"></div>
  <div class="deck" id="read"></div>
</section>

<div class="bar"><span>Chart window</span><button data-r="5Y" class="on">5Y</button><button data-r="10Y">10Y</button><button data-r="MAX">Max</button></div>

<section class="card">
  <p class="eyebrow">Breadth components · score 0–6</p>
  <p class="deck" id="deck1"></p>
  <h3>Breadth Components score and the S&amp;P 500</h3>
  <div class="chartbox" style="height:500px"><canvas id="c1"></canvas></div>
  <div class="legendrow" id="lg1"></div>
  <p class="cap" id="cap1"></p>
</section>

<section class="card">
  <p class="eyebrow">Breadth components · net reading</p>
  <p class="deck" id="deck2"></p>
  <h3>Breadth washouts vs divergences</h3>
  <div class="chartbox" style="height:500px"><canvas id="c2"></canvas></div>
  <div class="legendrow" id="lg2"></div>
  <p class="cap" id="cap2"></p>
</section>

<section class="card">
  <p class="eyebrow">Breadth thrusts · aggregator</p>
  <p class="deck" id="deck3"></p>
  <h3>Breadth Thrust Aggregator</h3>
  <div class="chartbox" style="height:560px"><canvas id="c3"></canvas></div>
  <div class="legendrow" id="lg3"></div>
  <p class="cap" id="cap3"></p>
</section>

<section class="card">
  <p class="eyebrow">Inputs · constituent breadth</p>
  <p class="deck" id="deck4"></p>
  <h3>The four inputs</h3>
  <div class="chartbox" style="height:620px"><canvas id="c4"></canvas></div>
  <p class="cap" id="cap4"></p>
</section>

<section class="card">
  <p class="eyebrow">Evidence · thrust selection</p>
  <p class="deck" id="subT1"></p>
  <h3>Which thrusts earn a place in the model</h3>
  <div class="tblwrap"><table id="t1"></table></div>
  <p class="cap" id="fnT1"></p>
</section>

<div class="cols">
  <section class="card">
    <p class="eyebrow">Evidence · thrust state</p>
    <p class="deck">Average S&amp;P 500 forward price return from every day spent in each state.</p>
    <h3>Does the thrust state matter?</h3>
    <div class="tblwrap"><table id="t2"></table></div>
    <p class="cap" id="fnT2"></p>
  </section>
  <section class="card">
    <p class="eyebrow">Evidence · component score</p>
    <p class="deck">The same test, by Breadth Components score.</p>
    <h3>Does the component score matter?</h3>
    <div class="tblwrap"><table id="t3"></table></div>
    <p class="cap" id="fnT3"></p>
  </section>
</div>

<section class="card">
  <p class="eyebrow">Evidence · component legs</p>
  <p class="deck">Forward returns from the days each leg’s trigger condition is true. Divergences are compared with all days the S&amp;P is within 3% of its 52-week high; washouts with all days.</p>
  <h3>Do the individual component legs carry information?</h3>
  <div class="tblwrap"><table id="t4"></table></div>
  <p class="cap" id="fnT4"></p>
</section>

<section class="card">
  <p class="eyebrow">Signal log</p>
  <p class="deck">First firing in each 21-session cluster, model signals only. Forward returns are S&amp;P 500 price returns from the signal close.</p>
  <h3>Recent thrust signals</h3>
  <div class="tblwrap"><table id="t5"></table></div>
</section>

<section class="card">
  <p class="eyebrow">Method and caveats</p>
  <ul class="notes" id="notes"></ul>
</section>
</div>
<div class="footbar"><div class="in"><span id="fb_date"></span><span>S&amp;P 500 Breadth Aggregators</span><span>Breadth Components · Thrust Aggregator</span></div></div>

<script>
const P = /*@DATA*/null/*@END*/;
// Deck palette: navy, yellow, grey, green, blue; orange for overlays. One typeface throughout.
const K = {ink:'#000000', navy:'#1F4E79', yellow:'#FFC000', grey:'#A5A5A5', green:'#70AD47', blue:'#4472C4', accent:'#ED7D31', mute:'#808080', grid:'#BDD7EE', pale:'#DEEBF7',
  up:'#548235', dn:'#C00000', yellowTxt:'#BF9000',
  spx:'#000000', divA:'#ED7D31', divB:'#4472C4', divC:'#A5A5A5', w1:'#FFC000', w2:'#1F4E79', count:'#1F4E79',
  bandBad:'#C0000021', bandGood:'#70AD4738'};
const FONT = "Bahnschrift, 'Barlow', 'Segoe UI', Arial, sans-serif", MONO = FONT, SANS = FONT;
Chart.defaults.font.family = FONT; Chart.defaults.font.size = 12; Chart.defaults.font.weight = 300; Chart.defaults.color = K.mute; Chart.defaults.animation = false;
Chart.defaults.scale.border.display = false;

// ---------- formatting ----------
const MIN = '−';
const pfmt = p => p == null ? '–' : p < 0.001 ? '<0.001' : p.toFixed(3);
const sgn = (v, d=1) => v == null ? '–' : (v < 0 ? MIN : '+') + Math.abs(v).toFixed(d);
const pc = (v, d=1) => v == null ? '–' : sgn(v, d) + '%';
const num = (v, d=0) => v == null ? '–' : (v < 0 ? MIN : '') + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const dfmt = s => s ? (+s.slice(8,10)) + ' ' + MON[+s.slice(5,7)-1] + ' ' + s.slice(0,4) : '–';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ---------- header, tiles, read ----------
const N = P.now, M = P.meta, B = P.base.all;
const longDate = s => (+s.slice(8,10)) + ' ' + ['January','February','March','April','May','June','July','August','September','October','November','December'][+s.slice(5,7)-1] + ' ' + s.slice(0,4);
$('meta').innerHTML = '<span>Snapshot&nbsp; <b>' + esc(M.built) + '</b></span><span>Data to&nbsp; <b>' + dfmt(M.asof) + ' close</b></span><span>Live&nbsp; <b>Yahoo Finance prices (live)</b> · <b>index membership (fja05680 + Wikipedia, live)</b></span><span>Universe&nbsp; <b>' + M.members_last + ' S&amp;P 500 members, point-in-time</b></span><span>Evidence&nbsp; <b>' + M.evid_start.slice(0,4) + ' to date · ' + num(M.n_days) + ' sessions</b></span>';
$('fb_date').textContent = longDate(M.asof);
const nDiv = N.dA + N.dB + N.dC;
const stName = {1:'Bullish', 0:'Neutral', '-1':'No thrust'};
const stCls = {1:'good', 0:'neu', '-1':'bad'};
const scoreCls = N.score === 0 ? 'bad' : N.score >= 4 ? 'good' : 'neu';
const scoreTxt = N.score === 0 ? 'All three divergences' : N.score >= 4 ? 'Washout' : nDiv ? nDiv + ' divergence' + (nDiv > 1 ? 's' : '') + ' active' : 'Clean tape';
$('kpis').innerHTML = [
  ['Breadth Components', N.score + '<small> / 6</small>', '<span class="pill ' + scoreCls + '">' + scoreTxt + '</span> net ' + sgn(N.net, 0) + ' since ' + dfmt(N.net_since)],
  ['Thrust Aggregator', N.tcount + '<small> active</small>', '<span class="pill ' + stCls[N.tstate] + '">' + stName[N.tstate] + ' (' + sgn(N.tstate, 0) + ')</span> since ' + dfmt(N.tstate_since)],
  ['% above 50DMA · 200DMA', num(N.p50, 0) + '<small>%</small> · ' + num(N.p200, 0) + '<small>%</small>', '%>200DMA 52-week peak ' + num(N.p200max, 0) + '%'],
  ['McClellan Osc · Summation', num(N.osc, 0) + ' · ' + num(N.msi, 0), 'S&amp;P 500 ' + num(N.spx, 0) + ', ' + pc(N.dd) + ' vs 52-week high'],
].map((k, i) => '<div class="kpi' + (i < 2 ? ' now' : '') + '"><div class="lab">' + k[0] + '</div><div class="val">' + k[1] + '</div><div class="note">' + k[2] + '</div></div>').join('');

(function read(){
  const b0 = P.buckets[0], b4 = P.buckets[P.buckets.length - 1], T = P.tst, lt = N.last_thrust;
  const names = []; if (N.dA) names.push('%>200DMA'); if (N.dB) names.push('Summation Index'); if (N.dC) names.push('%>50DMA');
  let p1;
  if (N.score === 0) p1 = 'All three divergence legs are active and no washout offsets them, the configuration Macro Ops flags as a top warning. ';
  else if (N.score >= 4) p1 = 'A washout is in force (oversold ' + (N.w1 ? 'on' : 'off') + ', deep oversold ' + (N.w2 ? 'on' : 'off') + ')' + (nDiv ? ', partly offset by ' + nDiv + ' active divergence leg' + (nDiv > 1 ? 's' : '') + ' (' + names.join(', ') + ')' : '') + '. ';
  else p1 = (nDiv ? nDiv + ' of 3 divergence legs active (' + names.join(', ') + ')' : 'No divergence leg is active') + (N.w1 || N.w2 ? ', with a washout flag on' : ', with no washout') + '. ';
  p1 = '<b>Components: ' + N.score + '/6, net ' + sgn(N.net, 0) + ' since ' + dfmt(N.net_since) + '.</b> ' + p1 +
    'Since ' + M.evid_start.slice(0,4) + ', days at score 0 were followed by an average ' + pc(b0.all['12m']) + ' over 12 months (higher ' + b0.all.w12m + '% of the time) against ' + pc(B['12m']) + ' (' + B.w12m + '%) for all days; washout days (score 4+) by ' + pc(b4.all['12m']) + '. ' +
    (b0.h2['12m'] > 0 && b0.h2['12m'] < P.base.h2['12m'] ? 'From ' + M.split + ' the score-0 penalty is a smaller gain, not a loss: ' : 'From ' + M.split + ': ') + pc(b0.h2['12m']) + ' against ' + pc(P.base.h2['12m']) + ' for all days.';
  let p2 = '<b>Thrusts: ' + N.tcount + ' of ' + P.sel.length + ' active, state ' + stName[N.tstate].toLowerCase() + ' since ' + dfmt(N.tstate_since) + '.</b> ';
  p2 += lt ? 'The last model signal was the ' + esc(lt.s) + ' on ' + dfmt(lt.d) + '. ' : '';
  p2 += 'Bullish days (21-session high of 2+ active signals) were followed by ' + pc(T.bull.all['12m']) + ' over 12 months (higher ' + T.bull.all.w12m + '%); no-thrust days by ' + pc(T.bear.all['12m']) + ' (' + T.bear.all.w12m + '%). ' +
    (T.bear.h1['12m'] < 0 && T.bear.h2['12m'] > 0 ? 'The no-thrust state is negative only before ' + M.split + ' (' + pc(T.bear.h1['12m']) + '); since then it has meant lower returns, ' : 'By period, no-thrust days returned ' + pc(T.bear.h1['12m']) + ' before ' + M.split + ' and since then ') + pc(T.bear.h2['12m']) + ' against ' + pc(T.bull.h2['12m']) + ' in the bullish state.';
  $('read').innerHTML = '<p>' + p1 + '</p><p>' + p2 + '</p>';
})();

$('deck1').innerHTML = 'Each divergence leg adds +1 while it is absent; an oversold washout adds +1 and a deep oversold washout +2, so a clean tape with no washout scores 3. Shading marks score 0 (every divergence active) and score 4+ (washout). <b>Latest: ' + N.score + '/6, ' + scoreTxt.toLowerCase() + (N.score === 0 ? ' active' : '') + ', since ' + dfmt(N.net_since) + '.</b>';
$('deck2').innerHTML = 'The same legs as a balance: oversold washouts stack above zero, active negative breadth divergences below it, and the line is the net reading (score − 3). <b>Latest: net ' + (N.net === 0 ? '0' : sgn(N.net, 0)) + ' since ' + dfmt(N.net_since) + '.</b>';
$('deck3').innerHTML = 'Count of active thrust signals out of ' + P.sel.length + '; each stays on for ' + M.hold_thrust + ' sessions after it fires. Bullish when the 21-session high of the count is 2 or more; no thrust when nothing has been active for 21 sessions. <b>Latest: ' + N.tcount + ' active, ' + stName[N.tstate].toLowerCase() + ' since ' + dfmt(N.tstate_since) + '.</b>';
$('deck4').innerHTML = 'Everything above is built from these four series, computed from S&amp;P 500 members only. Dashed lines are the thresholds the component legs and thrust signals use. <b>Latest: %>200DMA ' + num(N.p200, 0) + '%, %>50DMA ' + num(N.p50, 0) + '%, Oscillator ' + num(N.osc, 0) + ', Summation Index ' + num(N.msi, 0) + '.</b>';

$('cap1').innerHTML = 'Divergence legs register only while the S&amp;P is within 3% of its 52-week closing high and stay on for ' + M.hold_div + ' sessions after the last trigger day; the oversold washout stays on for ' + M.hold_w1 + ' sessions and the deep washout for ' + M.hold_w2 + '. <b>Score 0 = all three divergences active with no washout; score 4+ needs a washout.</b> Leg rules are in Method and caveats. Src: Yahoo Finance, own calculations.';
$('cap2').innerHTML = 'Net reading = score − 3, so it runs from −3 (three divergences, no washout) to +3 (both washouts, no divergence). <b>Washouts and divergences can be active together</b>, as in the weeks after a sharp fall from a high. Src: Yahoo Finance, own calculations.';
$('cap3').innerHTML = 'Bars in the middle strip show which of the ' + P.sel.length + ' model signals is active. Signal rules and the tests behind the selection are in the table below. <b>The aggregator stayed bullish through most of the 2022 bear market</b>, its clearest failure in the sample. Src: Yahoo Finance, own calculations.';
$('cap4').innerHTML = 'McClellan series are ratio-adjusted (net advances ÷ advances plus declines × 1000; 19- and 39-day EMAs); the Summation Index uses the closed form 19×EMA39 − 9×EMA19, so its level does not depend on the start date. <b>Not interchangeable with NYSE readings.</b> Src: Yahoo Finance, own calculations.';

// ---------- chart plumbing ----------
let RANGE = '5Y', i0 = 0, charts = [];
function startIdx(){
  const n = P.d.length; if (RANGE === 'MAX') return 0;
  const last = P.d[n-1], cut = (+last.slice(0,4) - (RANGE === '5Y' ? 5 : 10)) + last.slice(4);
  const i = P.d.findIndex(d => d >= cut); return i < 0 ? 0 : i;
}
const sl = a => a.slice(i0);
function spans(a, test){ const out = []; let s = -1; for (let i = 0; i < a.length; i++){ const t = test(a[i]); if (t && s < 0) s = i; if (!t && s >= 0){ out.push([s, i-1]); s = -1; } } if (s >= 0) out.push([s, a.length-1]); return out; }
function yearTicks(labels){
  const idx = []; for (let i = 1; i < labels.length; i++) if (labels[i].slice(0,4) !== labels[i-1].slice(0,4)) idx.push(i);
  const step = Math.max(1, Math.ceil(idx.length / 14)); return idx.filter((_, k) => (idx.length - 1 - k) % step === 0);
}
const NICE = [300,400,500,600,700,800,1000,1250,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12000,15000,20000];
const bandPlugin = {id:'bands', beforeDatasetsDraw(ch, _a, o){
  if (!o || !o.bands) return; const {ctx, chartArea:a, scales:{x}} = ch; const half = (a.right - a.left) / Math.max(1, ch.data.labels.length - 1) / 2;
  const top = o.axis ? ch.scales[o.axis].top : a.top, bot = o.axis ? ch.scales[o.axis].bottom : a.bottom;
  ctx.save(); o.bands.forEach(b => { ctx.fillStyle = b.color; b.spans.forEach(([s, e]) => { const x0 = Math.max(a.left, x.getPixelForValue(s) - half), x1 = Math.min(a.right, x.getPixelForValue(e) + half); ctx.fillRect(x0, top, x1 - x0, bot - top); }); }); ctx.restore(); }};
const crossPlugin = {id:'cross', afterDatasetsDraw(ch){ const t = ch.tooltip; if (!t || !t.getActiveElements().length) return; const {ctx, chartArea:a} = ch; const xp = t.getActiveElements()[0].element.x;
  ctx.save(); ctx.strokeStyle = '#1F4E7988'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xp, a.top); ctx.lineTo(xp, a.bottom); ctx.stroke(); ctx.restore(); }};
const endPlugin = {id:'endlab', afterDatasetsDraw(ch, _a, o){ if (!o || o.ds == null) return; const m = ch.getDatasetMeta(o.ds); if (!m.data.length) return; const p = m.data[m.data.length-1]; const {ctx} = ch;
  ctx.save(); ctx.font = "500 12px " + FONT; ctx.fillStyle = K.navy; ctx.textBaseline = 'middle'; ctx.fillText(o.text, p.x + 6, p.y); ctx.restore(); }};
const refPlugin = {id:'refs', afterDatasetsDraw(ch, _a, o){ if (!o) return; const {ctx, chartArea:a} = ch; ctx.save();
  const tag = (text, x, y, align, base, font, color) => { ctx.font = font; ctx.textAlign = align; ctx.textBaseline = base; const w = ctx.measureText(text).width, h = 14;
    const x0 = align === 'right' ? x - w - 3 : align === 'center' ? x - w / 2 - 3 : x - 3, y0 = base === 'bottom' ? y - h : base === 'top' ? y - 1 : y - h / 2;
    ctx.fillStyle = '#FFFFFFE6'; ctx.fillRect(x0, y0, w + 6, h + 1); ctx.fillStyle = color; ctx.fillText(text, x, y); };
  (o.lines || []).forEach(l => { const sc = ch.scales[l.axis]; if (l.v < sc.min || l.v > sc.max) return; const y = sc.getPixelForValue(l.v); ctx.strokeStyle = l.color; ctx.lineWidth = 1; ctx.setLineDash(l.solid ? [] : [5,4]); ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke(); ctx.setLineDash([]);
    if (l.text) tag(l.text, l.align === 'right' ? a.right - 5 : a.left + 6, y - 2, l.align || 'left', 'bottom', '300 11px ' + FONT, K.mute); });
  (o.texts || []).forEach(t => { const sc = ch.scales[t.axis]; tag(t.text, a.left + 8, t.top ? sc.top + 5 : sc.bottom - 5, 'left', t.top ? 'top' : 'bottom', '500 11px ' + FONT, t.color); });
  (o.titles || []).forEach(t => { const sc = ch.scales[t.axis]; tag(t.text, a.left + 6, sc.top + 3, 'left', 'top', '400 12.5px ' + FONT, K.mute); });
  (o.marks || []).forEach(m => { const sc = ch.scales[m.axis]; ctx.font = '500 12px ' + FONT; ctx.fillStyle = K.navy; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(m.text, a.right + 6, sc.getPixelForValue(m.v)); });
  ctx.restore(); }};
const lanePlugin = {id:'lanes', beforeDatasetsDraw(ch, _a, o){ if (!o || !o.lanes) return; const {ctx, chartArea:a, scales:{x}} = ch; const sc = ch.scales[o.axis]; const half = (a.right - a.left) / Math.max(1, ch.data.labels.length - 1) / 2;
  const n = o.lanes.length, h = (sc.bottom - sc.top) / n; ctx.save();
  o.lanes.forEach((ln, k) => { const y = sc.top + k * h; ctx.fillStyle = K.pale; ctx.fillRect(a.left, y + h * .5 - .5, a.right - a.left, 1); ctx.fillStyle = ln.color;
    ln.spans.forEach(([s, e]) => { const x0 = Math.max(a.left, x.getPixelForValue(s) - half), x1 = Math.min(a.right, x.getPixelForValue(e) + half); ctx.fillRect(x0, y + h * .2, Math.max(1.5, x1 - x0), h * .6); });
    ctx.font = '300 11.5px ' + FONT; ctx.fillStyle = K.mute; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(ln.label, a.left - 8, y + h / 2); });
  ctx.restore(); }};
Chart.register(bandPlugin, crossPlugin, endPlugin, refPlugin, lanePlugin);

const GRID = {color:K.grid, drawTicks:false, lineWidth:1};
function xAxis(labels){ const yt = yearTicks(labels); return {type:'category', grid:{display:false}, border:{display:false}, ticks:{autoSkip:false, maxRotation:0, color:K.mute, padding:6, font:{size:12}, callback(v){ return labels[v] ? labels[v].slice(0,4) : ''; }}, afterBuildTicks(ax){ ax.ticks = yt.map(i => ({value:i})); }}; }
function priceAxis(px, weight){ const v = px.filter(x => x != null), lo = Math.min(...v) * .96, hi = Math.max(...v) * 1.04;
  return {type:'logarithmic', position:'left', stack:'s', stackWeight:weight, min:lo, max:hi, grid:GRID, border:{display:false}, title:{display:true, text:'S&P 500 (log)', color:K.mute, font:{size:11}},
    ticks:{color:K.navy, padding:6, font:{size:11.5}, callback:v => num(v)}, afterBuildTicks(ax){ let t = NICE.filter(n => n >= lo && n <= hi); if (t.length > 8) t = t.filter((_, k) => k % 2 === (t.length - 1) % 2); ax.ticks = t.map(n => ({value:n})); }}; }
function panelAxis(o){ const base = {type:'linear', position:'left', stack:'s', stackWeight:1, grid:GRID, border:{display:false}, ticks:{color:K.navy, padding:6, font:{size:11.5}, autoSkip:false}};
  const out = Object.assign({}, base, o); out.ticks = Object.assign({}, base.ticks, o.ticks || {}); if (out.title) out.title = Object.assign({color:K.mute}, out.title); return out; }
const force = list => function(ax){ ax.ticks = list.map(v => ({value:v})); };
const area = (label, data, color, extra) => Object.assign({type:'line', label, data, backgroundColor:color, borderColor:color, borderWidth:0, pointRadius:0, pointHoverRadius:0, stepped:true, fill:true}, extra);
const line = (label, data, color, extra) => Object.assign({type:'line', label, data, borderColor:color, backgroundColor:color, borderWidth:1.6, pointRadius:0, pointHoverRadius:3, fill:false, tension:0}, extra);
const tipBase = {mode:'index', intersect:false, backgroundColor:K.navy, titleColor:'#FFFFFF', bodyColor:'#FFFFFF', footerColor:'#FFFFFF', cornerRadius:0, titleFont:{family:FONT, weight:500, size:12.5}, bodyFont:{family:FONT, weight:300, size:12}, footerFont:{family:FONT, size:12, weight:500}, padding:10, boxWidth:9, boxHeight:9, usePointStyle:false, displayColors:true};
const swatch = c => { const col = c.dataset.borderWidth ? c.dataset.borderColor : c.dataset.backgroundColor; const v = col === K.navy || col === K.ink || col === '#1F4E79D9' ? '#FFFFFF' : col; return {borderColor:v, backgroundColor:v}; };
function legend(id, items){ $(id).innerHTML = items.map(it => '<span>' + (it.ln ? '<span style="display:inline-block;width:14px;height:' + (it.thin ? 2 : 3) + 'px;background:' + it.c + ';vertical-align:middle;margin-right:6px"></span>'
  : '<span class="dot sq" style="background:' + it.c + (it.band ? ';border:1px solid #D9D9D9;box-sizing:border-box' : '') + '"></span>') + it.t + '</span>').join(''); }
function mk(id, cfg){ const c = new Chart($(id), cfg); charts.push(c); return c; }
const layout = {padding:{right:58, left:4, top:4}};
const common = (labels, scales, plugins) => ({maintainAspectRatio:false, responsive:true, layout, interaction:{mode:'index', intersect:false}, scales:Object.assign({x:xAxis(labels)}, scales), plugins:Object.assign({legend:{display:false}}, plugins)});

function draw(){
  charts.forEach(c => c.destroy()); charts = []; i0 = startIdx();
  const L = sl(P.d), px = sl(P.spx), score = sl(P.score), dA = sl(P.dA), dB = sl(P.dB), dC = sl(P.dC), w1 = sl(P.w1), w2 = sl(P.w2);
  const endTxt = num(px[px.length-1]);
  const titleCb = items => dfmt(L[items[0].dataIndex]);

  // 1. components
  legend('lg1', [{c:K.divA, t:'Divergence A absent · %>200DMA (+1)'}, {c:K.divB, t:'Divergence B absent · Summation Index (+1)'}, {c:K.divC, t:'Divergence C absent · %>50DMA (+1)'}, {c:K.w1, t:'Oversold washout (+1)'}, {c:K.w2, t:'Deep oversold washout (+2)'},
    {c:K.spx, t:'S&amp;P 500', ln:1, thin:1}, {c:K.bandBad, t:'Score 0', band:1}, {c:K.bandGood, t:'Score 4+', band:1}]);
  mk('c1', {data:{labels:L, datasets:[
      line('S&P 500', px, K.spx, {yAxisID:'yP', borderWidth:1.4}),
      area('Divergence A absent', dA.map(v => 1 - v), K.divA, {yAxisID:'yS', fill:'origin'}),
      area('Divergence B absent', dB.map(v => 1 - v), K.divB, {yAxisID:'yS', fill:'-1'}),
      area('Divergence C absent', dC.map(v => 1 - v), K.divC, {yAxisID:'yS', fill:'-1'}),
      area('Oversold washout', w1, K.w1, {yAxisID:'yS', fill:'-1'}),
      area('Deep oversold washout', w2.map(v => 2 * v), K.w2, {yAxisID:'yS', fill:'-1'})]},
    options:common(L, {yS:panelAxis({stacked:true, min:0, max:6.7, stackWeight:2, title:{display:true, text:'Score', font:{size:11}}, afterBuildTicks:force([0,1,2,3,4,5,6])}), yP:priceAxis(px, 3)},
      {bands:{bands:[{color:K.bandBad, spans:spans(score, v => v === 0)}, {color:K.bandGood, spans:spans(score, v => v >= 4)}]}, endlab:{ds:0, text:endTxt}, refs:{marks:[{axis:'yS', v:Math.max(.25, score[score.length-1]), text:'Now ' + score[score.length-1]}]},
       tooltip:Object.assign({}, tipBase, {filter:it => it.datasetIndex === 0 || it.raw > 0, callbacks:{labelColor:swatch, title:titleCb, label:c => c.datasetIndex === 0 ? ' S&P 500  ' + num(c.raw, 2) : ' ' + c.dataset.label + '  +' + c.raw, footer:it => 'Score ' + score[it[0].dataIndex] + ' / 6'}})})});

  // 2. net
  legend('lg2', [{c:K.w2, t:'Deep oversold washout (+2)'}, {c:K.w1, t:'Oversold washout (+1)'}, {c:K.divA, t:'Divergence A · %>200DMA (' + MIN + '1)'}, {c:K.divB, t:'Divergence B · Summation Index (' + MIN + '1)'}, {c:K.divC, t:'Divergence C · %>50DMA (' + MIN + '1)'}, {c:K.spx, t:'Net reading (score ' + MIN + ' 3)', ln:1, thin:1}, {c:K.spx, t:'S&amp;P 500', ln:1, thin:1}]);
  const net = score.map(v => v - 3);
  mk('c2', {data:{labels:L, datasets:[
      line('S&P 500', px, K.spx, {yAxisID:'yP', borderWidth:1.4}),
      line('Net reading', net, K.spx, {yAxisID:'yS', stepped:true, borderWidth:1.5, stack:'net'}),
      area('Deep oversold washout', w2.map(v => 2 * v), K.w2, {yAxisID:'yS', stack:'pos', fill:'origin'}),
      area('Oversold washout', w1, K.w1, {yAxisID:'yS', stack:'pos', fill:'-1'}),
      area('Divergence A', dA.map(v => -v), K.divA, {yAxisID:'yS', stack:'neg', fill:'origin'}),
      area('Divergence B', dB.map(v => -v), K.divB, {yAxisID:'yS', stack:'neg', fill:'-1'}),
      area('Divergence C', dC.map(v => -v), K.divC, {yAxisID:'yS', stack:'neg', fill:'-1'})]},
    options:common(L, {yS:panelAxis({stacked:true, min:-3.8, max:3.8, stackWeight:2, title:{display:true, text:'Net', font:{size:11}}, afterBuildTicks:force([-3,-2,-1,0,1,2,3]), ticks:{autoSkip:false, callback:v => v === 0 ? '0' : sgn(v, 0)}}), yP:priceAxis(px, 3)},
      {endlab:{ds:0, text:endTxt}, refs:{lines:[{axis:'yS', v:0, color:K.ink, solid:1}], texts:[{axis:'yS', top:1, text:'BULLISH · washout', color:K.up}, {axis:'yS', text:'BEARISH · divergence', color:K.dn}], marks:[{axis:'yS', v:net[net.length-1], text:'Now ' + (net[net.length-1] === 0 ? '0' : sgn(net[net.length-1], 0))}]},
       tooltip:Object.assign({}, tipBase, {filter:it => it.datasetIndex === 0 || (it.datasetIndex > 1 && it.raw !== 0), callbacks:{labelColor:swatch, title:titleCb, label:c => c.datasetIndex === 0 ? ' S&P 500  ' + num(c.raw, 2) : ' ' + c.dataset.label + '  ' + sgn(c.raw, 0), footer:it => 'Net ' + sgn(net[it[0].dataIndex], 0).replace('+0', '0')}})})});

  // 3. thrust
  const tc = sl(P.tcount), ts = sl(P.tstate), laneCol = [K.navy, K.yellow, K.grey, K.accent, K.blue];
  const short = {zbt:'Zweig', adt:'Whaley A/D', dg:'deGraaf', p50:'%>50DMA >90', osc:'McClellan Osc', bam:'Deemer'};
  legend('lg3', [{c:K.count, t:'Active thrust signals (count)'}, {c:K.spx, t:'S&amp;P 500', ln:1, thin:1}, {c:K.bandGood, t:'Bullish (+1): 21-session high of 2 or more', band:1}, {c:K.bandBad, t:'No thrust (' + MIN + '1): none active in 21 sessions', band:1}]);
  mk('c3', {data:{labels:L, datasets:[
      line('S&P 500', px, K.spx, {yAxisID:'yP', borderWidth:1.4}),
      area('Active thrust signals', tc, '#1F4E79D9', {yAxisID:'yC', borderColor:K.navy, borderWidth:1})]},
    options:Object.assign(common(L, {yC:panelAxis({min:0, max:P.sel.length + .4, stackWeight:2, title:{display:true, text:'Count', font:{size:11}}, afterBuildTicks:force(P.sel.map((_, k) => k).concat(P.sel.length))}),
        yL:panelAxis({min:0, max:1, stackWeight:1.5, grid:{display:false}, ticks:{display:false}, title:{display:false}}), yP:priceAxis(px, 4)},
      {bands:{bands:[{color:K.bandGood, spans:spans(ts, v => v === 1)}, {color:K.bandBad, spans:spans(ts, v => v === -1)}]}, endlab:{ds:0, text:endTxt},
       lanes:{axis:'yL', lanes:P.sel.map((s, k) => ({label:short[s.key] || s.name, color:laneCol[k % laneCol.length], spans:spans(sl(P.act[s.key]), v => v === 1)}))},
       refs:{lines:[{axis:'yC', v:2, color:K.accent, text:'Bullish at 2+', align:'right'}], marks:[{axis:'yC', v:Math.max(.2, tc[tc.length-1]), text:'Now ' + tc[tc.length-1]}]},
       tooltip:Object.assign({}, tipBase, {callbacks:{labelColor:swatch, title:titleCb, label:c => c.datasetIndex === 0 ? ' S&P 500  ' + num(c.raw, 2) : ' Active signals  ' + c.raw,
         footer:it => { const i = it[0].dataIndex, on = P.sel.filter(s => P.act[s.key][i0 + i] === 1).map(s => short[s.key]); return (on.length ? on.join(', ') + '\n' : '') + 'State ' + stName[ts[i]] + ' (' + sgn(ts[i], 0).replace('+0', '0') + ')'; }}})}), {layout:{padding:{right:58, left:66, top:4}}})});

  // 4. inputs
  const rng = (a, step, floor) => { const v = a.filter(x => x != null), lo = Math.min(-floor, ...v), hi = Math.max(floor, ...v), r = hi - lo, t = [];
    for (let k = Math.ceil(lo / step) * step; k <= hi; k += step) t.push(k);
    return panelAxis({min:lo - .06 * r, max:hi + .3 * r, afterBuildTicks:force(t.length > 5 ? t.filter(k => k % (2 * step) === 0) : t), ticks:{autoSkip:false, callback:v => num(v)}}); };
  const p50 = sl(P.p50), p200 = sl(P.p200), osc = sl(P.osc), msi = sl(P.msi);
  mk('c4', {data:{labels:L, datasets:[
      line('Summation Index', msi, K.navy, {yAxisID:'y4', borderWidth:1.4}),
      line('McClellan Oscillator', osc, K.navy, {yAxisID:'y3', borderWidth:1.1}),
      line('% above 50DMA', p50, K.navy, {yAxisID:'y2', borderWidth:1.2}),
      line('% above 200DMA', p200, K.navy, {yAxisID:'y1', borderWidth:1.4})]},
    options:common(L, {
        y4:rng(msi, 500, 500), y3:rng(osc, 100, 100),
        y2:panelAxis({min:-6, max:128, afterBuildTicks:force([0,25,50,75,100])}),
        y1:panelAxis({min:-6, max:128, afterBuildTicks:force([0,25,50,75,100])})},
      {refs:{lines:[{axis:'y1', v:20, color:K.navy, text:'20 · deep washout', align:'right'}, {axis:'y2', v:90, color:K.green, text:'90 · thrust', align:'right'}, {axis:'y2', v:50, color:K.accent, text:'50 · divergence C if S&P near high', align:'right'}, {axis:'y2', v:20, color:K.yellowTxt, text:'20 · washout'}, {axis:'y2', v:10, color:K.navy, text:'10 · deep washout', align:'right'},
          {axis:'y3', v:90, color:K.green, text:'+90 · thrust', align:'right'}, {axis:'y3', v:-80, color:K.yellowTxt, text:MIN + '80 · washout', align:'right'}, {axis:'y3', v:0, color:K.grey, solid:1},
          {axis:'y4', v:250, color:K.accent, text:'+250 · divergence B if S&P near high', align:'right'}, {axis:'y4', v:0, color:K.grey, solid:1}],
        titles:[{axis:'y1', text:'% of members above 200-day MA'}, {axis:'y2', text:'% of members above 50-day MA'}, {axis:'y3', text:'McClellan Oscillator (ratio-adjusted)'}, {axis:'y4', text:'McClellan Summation Index (ratio-adjusted)'}]},
       tooltip:Object.assign({}, tipBase, {itemSort:(a, b) => b.datasetIndex - a.datasetIndex, callbacks:{labelColor:swatch, title:titleCb, label:c => ' ' + c.dataset.label + '  ' + num(c.raw, c.datasetIndex === 0 ? 0 : 1)}})})});
}
document.querySelectorAll('.bar button').forEach(b => b.addEventListener('click', () => { RANGE = b.dataset.r; document.querySelectorAll('.bar button').forEach(x => x.classList.toggle('on', x === b)); draw(); }));

// ---------- tables ----------
(function tables(){
  const bl = P.base, T = P.tst, S = M.split;
  // t1: candidates
  const failTxt = {win:'12m win <90%', p:'placebo p ≥0.05', halves:'weak in one half', n:'fewer than 15 signals'};
  const row = c => { const fails = Object.keys(c.pass).filter(k => !c.pass[k]).map(k => failTxt[k]);
    return '<tr class="' + (c.selected ? 'sel' : 'rej') + '"><td class="t">' + esc(c.name) + '<small>' + esc(c.rule) + '</small></td><td>' + c.n + '</td><td>' + c.episodes + '</td><td>' + pc(c['3m_avg']) + '</td><td>' + pc(c['6m_avg']) + '</td><td>' + c['6m_win'] + '%</td><td>' + pc(c['12m_avg']) + '</td><td>' + c['12m_win'] + '%</td><td>' + pc(c.dd_worst) + '</td><td>' + pfmt(c.p12) + '</td><td>' + pc(c['12m_h1']) + '</td><td>' + pc(c['12m_h2']) + '</td><td class="t">' +
      (c.selected ? '<span class="pill number">In model</span>' + (fails.length ? '<small>now fails: ' + fails.join(', ') + '</small>' : '') : '<span class="pill none">Rejected</span><small>' + fails.join(', ') + '</small>') + '</td></tr>'; };
  const head = '<tr><th>Signal</th><th>Signals</th><th>Episodes</th><th>+3m</th><th>+6m</th><th>6m higher</th><th>+12m</th><th>12m higher</th><th>Worst 6m drawdown</th><th>Placebo p (12m)</th><th>+12m pre-' + S + '</th><th>+12m ' + S + '+</th><th style="text-align:left">Verdict</th></tr>';
  const baseRow = '<tr class="base"><td class="t">All days (baseline)</td><td>–</td><td>–</td><td>' + pc(bl.all['3m']) + '</td><td>' + pc(bl.all['6m']) + '</td><td>' + bl.all.w6m + '%</td><td>' + pc(bl.all['12m']) + '</td><td>' + bl.all.w12m + '%</td><td>–</td><td>–</td><td>' + pc(bl.h1['12m']) + '</td><td>' + pc(bl.h2['12m']) + '</td><td></td></tr>';
  const capRow = c => '<tr><td class="t">' + esc(c.name) + (c.rule ? '<small>' + esc(c.rule) + '</small>' : '') + '</td><td>' + c.n + '</td><td>' + c.episodes + '</td><td>' + pc(c['3m_avg']) + '</td><td>' + pc(c['6m_avg']) + '</td><td>' + c['6m_win'] + '%</td><td>' + pc(c['12m_avg']) + '</td><td>' + c['12m_win'] + '%</td><td>' + pc(c.dd_worst) + '</td><td>' + pfmt(c.p12) + '</td><td>' + pc(c['12m_h1']) + '</td><td>' + pc(c['12m_h2']) + '</td><td class="t"><small>feeds the washout legs, not the thrust count</small></td></tr>';
  const selC = P.cand.filter(c => c.selected), rejC = P.cand.filter(c => !c.selected);
  $('t1').innerHTML = head + baseRow + '<tr class="grp"><td colspan="13">Thrust signals in the model</td></tr>' + selC.map(row).join('') + '<tr class="grp"><td colspan="13">Thrust signals tested and rejected</td></tr>' + rejC.map(row).join('') + '<tr class="grp"><td colspan="13">Capitulation signals (reference)</td></tr>' + P.caps.map(capRow).join('');
  $('subT1').textContent = P.cand.length + ' published thrust definitions computed on S&P 500 members since ' + M.evid_start.slice(0,4) + ' and scored on S&P 500 forward price returns. A signal is kept only if it clears all four tests: 12-month hit rate of 90%+, placebo p below 0.05, 12-month average at least 3 points above baseline in both halves of the sample, and at least 15 signals.';
  $('fnT1').textContent = 'Signals: first firing in each 21-session cluster. Episodes: clusters more than 126 sessions apart, the closer guide to independent observations. Placebo p: share of all circular date-shifts of the same signal set that produce a 12-month average at least as high. Worst 6m drawdown: lowest close within 126 sessions of any signal, against the signal close.';

  // t2: thrust states
  const srow = (lab, o, cls) => '<tr' + (cls ? ' class="' + cls + '"' : '') + '><td class="t">' + lab + '</td><td>' + num(o.all.pct, 0) + '%</td><td>' + pc(o.all['3m']) + '</td><td>' + pc(o.all['6m']) + '</td><td>' + pc(o.all['12m']) + '</td><td>' + o.all.w12m + '%</td><td>' + pc(o.h1['12m']) + '</td><td>' + pc(o.h2['12m']) + '</td></tr>';
  const shead = '<tr><th>State</th><th>Share of days</th><th>+3m</th><th>+6m</th><th>+12m</th><th>12m higher</th><th>+12m pre-' + S + '</th><th>+12m ' + S + '+</th></tr>';
  const allB = {all:bl.all, h1:bl.h1, h2:bl.h2};
  $('t2').innerHTML = shead + srow('Bullish (+1)', T.bull) + srow('Neutral (0)', T.neutral) + srow('No thrust (' + MIN + '1)', T.bear) + srow('All days', allB, 'base') +
    '<tr class="grp"><td colspan="8">By count of active signals (full sample)</td></tr>' + P.bycount.map(c => '<tr><td class="t">' + c.count + ' active</td><td>' + num(c.pct, 0) + '%</td><td>' + pc(c['3m']) + '</td><td>' + pc(c['6m']) + '</td><td>' + pc(c['12m']) + '</td><td>' + c.w12m + '%</td><td>–</td><td>–</td></tr>').join('');
  $('fnT2').textContent = 'Next-day annualised S&P 500 return by state: bullish ' + pc(T.bull.ann) + ', neutral ' + pc(T.neutral.ann) + ', no thrust ' + pc(T.bear.ann) + ' (annualised volatility ' + T.bull.vol + '%, ' + T.neutral.vol + '%, ' + T.bear.vol + '%). The pre-' + S + ' no-thrust loss comes from the 2000–02 and 2008 bear markets.';

  // t3: score buckets
  $('t3').innerHTML = shead.replace('State', 'Score') + P.buckets.map(x => srow(esc(x.label), x)).join('') + srow('All days', allB, 'base');
  $('fnT3').textContent = 'The score is not monotonic. Divergence legs can only register while the S&P is within 3% of a 52-week high, so score 3 also covers bear-market stretches between washouts, which is why its pre-' + S + ' return is poor.';

  // t4: legs
  const nb = P.near_base;
  const lhead = '<tr><th>Leg</th><th>Share of days</th><th>+1m</th><th>+3m</th><th>+6m</th><th>6m higher</th><th>+12m</th><th>12m higher</th><th>+12m pre-' + S + '</th><th>+12m ' + S + '+</th></tr>';
  const lrow = (name, rule, o, cls) => '<tr' + (cls ? ' class="' + cls + '"' : '') + '><td class="t">' + esc(name) + (rule ? '<small>' + esc(rule) + '</small>' : '') + '</td><td>' + num(o.all.pct, 1) + '%</td><td>' + pc(o.all['1m']) + '</td><td>' + pc(o.all['3m']) + '</td><td>' + pc(o.all['6m']) + '</td><td>' + o.all.w6m + '%</td><td>' + pc(o.all['12m']) + '</td><td>' + o.all.w12m + '%</td><td>' + pc(o.h1['12m']) + '</td><td>' + pc(o.h2['12m']) + '</td></tr>';
  const divs = P.legs.filter(l => l.kind === 'div'), wsh = P.legs.filter(l => l.kind === 'wash');
  $('t4').innerHTML = lhead + '<tr class="grp"><td colspan="10">Divergence legs (trigger days)</td></tr>' + divs.map(l => lrow(l.name, l.rule, l.trig)).join('') + lrow('S&P within 3% of 52-week high (comparison)', '', nb, 'base') +
    '<tr class="grp"><td colspan="10">Washout legs (trigger days)</td></tr>' + wsh.map(l => lrow(l.name, l.rule, l.trig)).join('') + lrow('All days (comparison)', '', allB, 'base') +
    '<tr class="grp"><td colspan="10">Raw oversold levels (full sample, for reference)</td></tr>' + P.rawlv.map(r => '<tr><td class="t">' + esc(r.name) + '</td><td>' + num(r.pct, 1) + '%</td><td>' + pc(r['1m']) + '</td><td>' + pc(r['3m']) + '</td><td>' + pc(r['6m']) + '</td><td>' + r.w6m + '%</td><td>' + pc(r['12m']) + '</td><td>' + r.w12m + '%</td><td>–</td><td>–</td></tr>').join('');
  const pen = l => l.trig.h1['12m'] < nb.h1['12m'] && l.trig.h2['12m'] < nb.h2['12m'] && l.trig.all['6m'] < nb.all['6m'];
  const yes = divs.filter(pen), no = divs.filter(l => !pen(l));
  const nm = ls => (ls.length > 1 ? 'Divergences ' : 'Divergence ') + ls.map(l => l.key).join(ls.length > 2 ? ', ' : ' and ');
  $('fnT4').textContent = (yes.length ? nm(yes) + (yes.length > 1 ? ' trail' : ' trails') + ' the near-high comparison at 6 months and, in both halves of the sample, at 12 months (12-month: ' + yes.map(l => l.key + ' ' + pc(l.trig.all['12m'])).join(', ') + ' against ' + pc(nb.all['12m']) + '). ' : 'No divergence leg trails the near-high comparison consistently. ') +
    (no.length ? nm(no) + (no.length > 1 ? ' have' : ' has') + ' no dependable standalone edge (6-month: ' + no.map(l => l.key + ' ' + pc(l.trig.all['6m'])).join(', ') + ' against ' + pc(nb.all['6m']) + '), the profile of a pullback inside an uptrend. ' : '') +
    'All three legs are kept at textbook levels: when this build was designed, tighter thresholds for B and C made the three-leg score-0 state less informative, not more.';

  // t5: log
  $('t5').innerHTML = '<tr><th>Date</th><th style="text-align:left">Signal</th><th>S&amp;P 500</th><th>+3m</th><th>+6m</th><th>+12m</th></tr>' + P.log.slice(0, 18).map(r => '<tr><td class="t" style="white-space:nowrap">' + dfmt(r.d) + '</td><td class="t">' + esc(r.s) + '</td><td>' + num(r.px) + '</td><td>' + pc(r.f3) + '</td><td>' + pc(r.f6) + '</td><td>' + pc(r.f12) + '</td></tr>').join('');

  // notes
  const z = P.cand.find(c => c.key === 'zbt'), bam = P.cand.find(c => c.key === 'bam');
  const mem = M.membership || {};
  $('notes').innerHTML = [
    '<b>What this is.</b> Macro Ops does not publish the rules behind its Breadth Components or Breadth Thrust Aggregator. This build reproduces the published structure (three divergence legs worth +1 when absent, washouts worth +1 and +2, net reading = score − 3; thrust count with a bullish state at a 21-session high of 2+ and a bearish state when nothing has fired in 21 sessions). The legs, thresholds and thrust list are this build’s own.',
    '<b>Component legs.</b> Divergence A: S&amp;P within 3% of its 52-week closing high while %>200DMA is 20+ points below its own 52-week peak. B: same price condition with the Summation Index below +250. C: same with %>50DMA below 50%. Each stays active ' + M.hold_div + ' sessions after its last trigger day. Oversold washout: McClellan Oscillator ≤ −80 or %>50DMA ≤ 20%, held ' + M.hold_w1 + ' sessions. Deep oversold washout: %>50DMA ≤ 10% or %>200DMA ≤ 20%, held ' + M.hold_w2 + ' sessions.',
    '<b>Universe.</b> All breadth is computed from S&amp;P 500 members on point-in-time membership, not NYSE issues. Large caps move together more than the NYSE list, so thresholds calibrated on NYSE data fire more often here: the Zweig rule gives ' + z.n + ' signals since ' + M.evid_start.slice(0,4) + ' against roughly eight on NYSE data. Thresholds are left as published and judged on results.',
    '<b>Price coverage.</b> Yahoo Finance drops delisted and acquired names. Priced members cover ' + M.cov_first + '% of the index in ' + M.evid_start.slice(0,4) + ', ' + M.cov_2010 + '% in 2010, ' + M.cov_2020 + '% in 2020 and ' + M.cov_last + '% now, so pre-2010 breadth is survivorship-tinted (levels such as %>200DMA read slightly high; day-to-day advance/decline direction is less affected). A reused ticker can occasionally map an old member slot to a different company.',
    '<b>Membership.</b> Point-in-time lists from the fja05680/sp500 history file (to ' + esc(mem.hist_last || '') + '), rolled forward with Wikipedia’s current constituent table' + (mem.wiki_applied && mem.wiki_applied.length ? ' (' + mem.wiki_applied.length + ' additions since; removal dates are not published there, so one leaver is retired per joiner)' : '') + '.',
    '<b>McClellan.</b> Ratio-adjusted net advances ×1000, 19- and 39-day EMAs. The Summation Index uses the closed form 19×EMA39 − 9×EMA19, so its level does not depend on the start date; it is not the NYSE RASI and its levels are not interchangeable with it.',
    '<b>Volume signals.</b> Up/down volume is constituent dollar volume, not NYSE up/down share volume. Every volume-based thrust failed the tests on this proxy, which may reflect the proxy as much as the signals.',
    '<b>Evidence limits.</b> Forward returns are price-only and overlap heavily; ' + P.cand.length + ' definitions were tried and ' + P.sel.length + ' kept, so some selection bias is built in. The sample holds about a dozen independent bear-to-bull turns, and the signals cluster around the same ones. Deemer’s Breakaway Momentum did not clear the placebo test when the model was specified (p = ' + (bam ? pfmt(bam.p12) : '–') + ' on current data) and largely duplicates the Whaley A/D thrust, so it was left out. 2022 produced several thrusts that failed over 3–6 months.',
    '<b>Labels.</b> “Bearish” in the Macro Ops sense means no thrust on the books. From ' + S + ' that state returned ' + pc(T.bear.h2['12m']) + ' over 12 months against ' + pc(bl.h2['12m']) + ' for all days, so read it as the absence of a tailwind. Score 0 likewise marks a thinner cushion (' + pc(P.buckets[0].h2['12m']) + ' from ' + S + '), not a sell signal.',
    '<b>Sources.</b> Prices: Yahoo Finance daily closes (dividend-adjusted for moving averages and advance/decline; ^GSPC for the index). Definitions: Zweig, <i>Winning on Wall Street</i>; Whaley, “Planes, Trains and Automobiles” (2010 Dow Award); deGraaf / Renaissance Macro 20-day-high thrust; Deemer, Breakaway Momentum; Desmond, “Identifying Bear Market Bottoms and New Bull Markets” (2002 Dow Award); McClellan Financial Publications; Macro Ops HUD Breadth Guide.'
  ].map(t => '<li>' + t + '</li>').join('');
})();

draw();
// Redraw once the web fallback font has loaded (no-op where Bahnschrift is installed)
if (document.fonts && document.fonts.load) Promise.all(['300 12px Barlow', '400 12px Barlow', '500 12px Barlow'].map(f => document.fonts.load(f).catch(() => null))).then(() => charts.forEach(c => c.update('none')));
</script>
</body>
</html>
'''


# =============================================================== 5. build
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--refresh", action="store_true", help="ignore the price cache")
    a = ap.parse_args()
    out = a.out or "docs/index.html"
    if a.refresh and os.path.exists(os.path.join(CACHE, "prices.pkl")):
        os.remove(os.path.join(CACHE, "prices.pkl"))
    hist, notes = get_membership()
    log("membership snapshots", len(hist), "| rolled forward:", len(notes.get("wiki_applied", [])), "changes | wiki ok:", notes.get("wiki_ok"))
    universe = set().union(*hist[hist["date"] >= "1997-12-01"]["set"])
    px = get_prices(universe, must=set(hist["set"].iloc[-1]))
    log("priced tickers", px["adj"].shape[1], "of", len(universe), "| last close", px["spx"].index[-1].date())
    b, _ = build_breadth(px, hist)
    P = build_model(b, notes, px["adj"].shape[1])
    # ---- sanity gate: never overwrite a good page with a broken build ----
    problems = []
    if P["meta"]["cov_last"] < 90:
        problems.append(f"only {P['meta']['cov_last']}% of current members priced (Yahoo throttling?)")
    if P["meta"]["members_last"] < 450:
        problems.append(f"only {P['meta']['members_last']} members on the last session")
    if os.path.exists(out):
        prev = re.search(r'"asof":"(\d{4}-\d{2}-\d{2})"', open(out, encoding="utf-8").read())
        if prev and P["meta"]["asof"] < prev.group(1):
            problems.append(f"new data ends {P['meta']['asof']}, before the published {prev.group(1)}")
    if problems:
        log("NOT WRITING:", "; ".join(problems))
        sys.exit(1)
    data = json.dumps(P, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html, k = re.subn(r"/\*@DATA\*/.*?/\*@END\*/", lambda m: "/*@DATA*/" + data + "/*@END*/", TEMPLATE, count=1, flags=re.S)
    assert k == 1, "data placeholder not found"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    n = P["now"]
    log(f"wrote {out} ({len(html)/1e6:.2f} MB) | as of {P['meta']['asof']} | components {n['score']}/6 (net {n['net']:+d}) "
        f"| thrust count {n['tcount']} state {n['tstate']:+d}")


if __name__ == "__main__":
    main()
