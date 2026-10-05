#!/usr/bin/env python3
"""
Ranking predictive-power backtest (validates OLD ranking formula only).

Does NOT modify screener.py, index.html, or screener_data.json.
Does NOT invent buy/sell recommendations or optimize new weights.

Feature / sample window: 2026-08-01 through latest closed trading day.
Lookback bars are downloaded earlier so d200 / RSI / acc_dist are valid.
Labels need T+10, so the last ~10 trading days are features-only (not scored).
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Reuse helpers from screener WITHOUT editing screener.py
from screener import (
    classify_position,
    compute_acc_dist,
    compute_rsi,
    get_sector,
    get_sp500_tickers,
)

# --- constants ---
FEATURE_START = dt.date(2026, 8, 1)  # only emit/score from this date onward
DOWNLOAD_START = "2025-01-01"  # lookback for SMA200 / 220-day filter
HISTORY_DIR = Path("history")
PRICE_CACHE = Path("history/_price_cache.parquet")
REPORT_PATH = Path("backtest_report.md")
MIN_HISTORY_DAYS = 220
TOP_N = 20
FWD_DAYS = 10

try:
    import pyarrow  # noqa: F401

    USE_PARQUET = True
    FEATURE_EXT = ".parquet"
except ImportError:
    USE_PARQUET = False
    FEATURE_EXT = ".csv"


def feature_path(d: dt.date) -> Path:
    return HISTORY_DIR / f"features_{d.isoformat()}{FEATURE_EXT}"


def save_features(df: pd.DataFrame, d: dt.date) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = feature_path(d)
    if USE_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_features(d: dt.date) -> pd.DataFrame:
    path = feature_path(d)
    if USE_PARQUET:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def score_old(daily_change: float, d20: float, d50: float, d200: float) -> float:
    """OLD ranking formula (spec) — not the live screener score."""
    norm_d200 = min(max(d200, -15.0), 25.0)
    norm_d50 = min(max(d50, -10.0), 20.0)
    norm_d20 = min(max(d20, -8.0), 15.0)
    return (
        3.0 * daily_change
        + 0.8 * norm_d20
        + 0.3 * norm_d50
        + 0.1 * norm_d200
        - (3.0 if d20 >= 10 else 0.0)
    )


def _normalize_yf_multi(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV DataFrame with DatetimeIndex (date-normalized)}."""
    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance may use (ticker, field) or (field, ticker)
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if set(tickers).intersection(set(level0.unique())):
            # columns like ticker -> fields
            for t in tickers:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].copy()
                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[t] = df
        else:
            # columns like field -> ticker
            for t in tickers:
                try:
                    df = data.xs(t, axis=1, level=1).copy()
                except KeyError:
                    continue
                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[t] = df
    else:
        # single ticker
        df = data.dropna(subset=["Close"]).copy()
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[tickers[0]] = df
    return out


def download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Batch download daily bars; cache to disk for resume."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    end = (dt.date.today() + dt.timedelta(days=1)).isoformat()

    if PRICE_CACHE.exists() and USE_PARQUET:
        print(f"Loading price cache {PRICE_CACHE} ...")
        cached = pd.read_parquet(PRICE_CACHE)
        # cached is long format: date, ticker, Open, High, Low, Close, Volume
        by_t: dict[str, pd.DataFrame] = {}
        for t, g in cached.groupby("ticker"):
            g = g.set_index("date").sort_index()
            g.index = pd.to_datetime(g.index).normalize()
            by_t[t] = g[["Open", "High", "Low", "Close", "Volume"]]
        missing = [t for t in tickers if t not in by_t]
        if not missing:
            print(f"Price cache hit for {len(by_t)} tickers.")
            return by_t
        print(f"Cache missing {len(missing)} tickers; re-downloading all.")

    print(f"Downloading {len(tickers)} tickers from {DOWNLOAD_START} ...")
    # yfinance batch in chunks to reduce failures
    chunk_size = 80
    by_ticker: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"  chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} ({len(chunk)} symbols)")
        try:
            raw = yf.download(
                tickers=chunk,
                start=DOWNLOAD_START,
                end=end,
                interval="1d",
                group_by="ticker",
                threads=True,
                auto_adjust=True,
                progress=False,
            )
            part = _normalize_yf_multi(raw, chunk)
            by_ticker.update(part)
        except Exception as e:
            print(f"  chunk failed ({e}); trying one-by-one")
            for t in chunk:
                try:
                    raw = yf.download(
                        tickers=t,
                        start=DOWNLOAD_START,
                        end=end,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                    )
                    part = _normalize_yf_multi(raw, [t])
                    by_ticker.update(part)
                except Exception as e2:
                    print(f"    skip {t}: {e2}")

    print(f"Downloaded {len(by_ticker)} / {len(tickers)} tickers.")

    # save long cache
    if USE_PARQUET and by_ticker:
        rows = []
        for t, df in by_ticker.items():
            tmp = df.reset_index()
            date_col = tmp.columns[0]
            tmp = tmp.rename(columns={date_col: "date"})
            tmp["ticker"] = t
            rows.append(tmp[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]])
        long_df = pd.concat(rows, ignore_index=True)
        long_df.to_parquet(PRICE_CACHE, index=False)
        print(f"Wrote price cache {PRICE_CACHE} ({len(long_df)} rows).")

    return by_ticker


def compute_ticker_features_upto(df: pd.DataFrame, ticker: str, asof: pd.Timestamp) -> dict | None:
    """Features for one ticker at asof using only data on/before asof."""
    hist = df.loc[:asof]
    if len(hist) < MIN_HISTORY_DAYS:
        return None
    if asof not in hist.index:
        return None

    close_series = hist["Close"]
    current_price = float(close_series.iloc[-1])
    if len(close_series) < 2:
        return None
    prev_price = float(close_series.iloc[-2])
    if prev_price == 0:
        return None

    daily_change = round(((current_price - prev_price) / prev_price) * 100, 2)

    sma20 = float(close_series.rolling(20).mean().iloc[-1])
    if len(close_series) < 50 or pd.isna(sma20) or sma20 == 0:
        return None
    sma50 = float(close_series.rolling(50).mean().iloc[-1])
    sma200 = float(close_series.rolling(200).mean().iloc[-1])
    if any(pd.isna(x) or x == 0 for x in (sma50, sma200)):
        return None

    d20 = round(((current_price - sma20) / sma20) * 100, 1)
    d50 = round(((current_price - sma50) / sma50) * 100, 1)
    d200 = round(((current_price - sma200) / sma200) * 100, 1)

    rsi = compute_rsi(close_series, 14)
    position = classify_position(d20, rsi)
    acc_dist = compute_acc_dist(hist)

    # vol_ratio: T volume / mean of prior 20 days (exclude today) — match screener
    vol_ratio = 1.0
    dollar_vol_20 = 0.0
    try:
        if "Volume" in hist.columns:
            vol_series = hist["Volume"].dropna()
            if len(vol_series) >= 2:
                today_vol = float(vol_series.iloc[-1])
                prior = vol_series.iloc[:-1].tail(20)
                mean_prior = float(prior.mean()) if len(prior) > 0 else 0.0
                if mean_prior > 0 and today_vol == today_vol:
                    vol_ratio = today_vol / mean_prior
            dv = (hist["Close"] * hist["Volume"]).dropna().tail(20)
            if len(dv) > 0:
                dollar_vol_20 = float(dv.mean())
    except Exception:
        vol_ratio = 1.0
        dollar_vol_20 = 0.0

    return {
        "date": asof.date().isoformat(),
        "ticker": ticker,
        "sector": get_sector(ticker),
        "close": round(current_price, 4),
        "daily_change": daily_change,
        "d20": d20,
        "d50": d50,
        "d200": d200,
        "rsi": rsi,
        "acc_dist": acc_dist,
        "position": position,
        "vol_ratio": round(vol_ratio, 3),
        "dollar_vol_20": round(dollar_vol_20, 2),
    }


def add_sector_pct(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["sector"]].append(r)
    for members in groups.values():
        n = len(members)
        if n == 1:
            members[0]["sector_pct"] = 0.5
        else:
            ordered = sorted(members, key=lambda x: x["daily_change"])
            for i, m in enumerate(ordered):
                m["sector_pct"] = i / (n - 1)


def trading_dates_in_range(by_ticker: dict[str, pd.DataFrame], start: dt.date, end: dt.date) -> list[pd.Timestamp]:
    """Union of trading dates from SPY if available, else all tickers."""
    spy = by_ticker.get("SPY")
    if spy is not None and not spy.empty:
        idx = spy.index
    else:
        idx = pd.DatetimeIndex([])
        for df in by_ticker.values():
            idx = idx.union(df.index)
        idx = idx.sort_values()
    dates = [d for d in idx if start <= d.date() <= end]
    return dates


def build_features_for_day(by_ticker: dict[str, pd.DataFrame], asof: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, df in by_ticker.items():
        try:
            feat = compute_ticker_features_upto(df, ticker, asof)
            if feat is not None:
                rows.append(feat)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    add_sector_pct(rows)
    out = pd.DataFrame(rows)
    out = out.sort_values(["ticker"]).reset_index(drop=True)
    return out


def forward_excess(
    by_ticker: dict[str, pd.DataFrame],
    ticker: str,
    t_date: pd.Timestamp,
    spy: pd.DataFrame,
) -> float | None:
    """excess_10 using ticker's 10th subsequent close and SPY on those dates."""
    df = by_ticker.get(ticker)
    if df is None or t_date not in df.index:
        return None
    loc = df.index.get_loc(t_date)
    if isinstance(loc, slice):
        return None
    if isinstance(loc, np.ndarray):
        loc = int(loc.argmax())  # unlikely
    loc = int(loc)
    end_loc = loc + FWD_DAYS
    if end_loc >= len(df.index):
        return None
    t_end = df.index[end_loc]
    c0 = float(df["Close"].iloc[loc])
    c1 = float(df["Close"].iloc[end_loc])
    if c0 == 0 or pd.isna(c0) or pd.isna(c1):
        return None
    if t_date not in spy.index or t_end not in spy.index:
        return None
    s0 = float(spy.loc[t_date, "Close"])
    s1 = float(spy.loc[t_end, "Close"])
    if s0 == 0 or pd.isna(s0) or pd.isna(s1):
        return None
    fwd_10 = c1 / c0 - 1.0
    spy_10 = s1 / s0 - 1.0
    return fwd_10 - spy_10


def rank_top20(day_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Top 20 by score_col desc, ticker alpha ascending as tiebreak."""
    tmp = day_df.copy()
    tmp = tmp.sort_values([score_col, "ticker"], ascending=[False, True])
    return tmp.head(TOP_N)


def summarize_block(daily_stats: pd.DataFrame, label: str) -> dict:
    """Aggregate daily top20 metrics for a period."""
    n = len(daily_stats)
    if n == 0:
        return {
            "label": label,
            "n_days": 0,
            "old_med": None,
            "d1_med": None,
            "sec_med": None,
            "univ_med": None,
            "old_hit": None,
            "d1_hit": None,
            "sec_hit": None,
            "old_pos_pct": None,
            "d1_pos_pct": None,
            "sec_pos_pct": None,
            "diff_old_1d": None,
            "diff_old_sec": None,
            "diff_old_univ": None,
            "avg_names": None,
        }

    def med(col: str) -> float:
        return float(daily_stats[col].median())

    def mean(col: str) -> float:
        return float(daily_stats[col].mean())

    def pos_pct(col: str) -> float:
        return float((daily_stats[col] > 0).mean())

    return {
        "label": label,
        "n_days": n,
        "old_med": med("old_excess_med"),
        "d1_med": med("d1_excess_med"),
        "sec_med": med("sec_excess_med"),
        "univ_med": med("univ_excess_med"),
        "old_hit": mean("old_hit"),
        "d1_hit": mean("d1_hit"),
        "sec_hit": mean("sec_hit"),
        "old_pos_pct": pos_pct("old_excess_med"),
        "d1_pos_pct": pos_pct("d1_excess_med"),
        "sec_pos_pct": pos_pct("sec_excess_med"),
        "diff_old_1d": med("diff_old_1d"),
        "diff_old_sec": med("diff_old_sec"),
        "diff_old_univ": med("diff_old_univ"),
        "avg_names": mean("n_names"),
    }


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def conclusion_sentence(full: dict) -> str:
    """Observational language only, per spec."""
    old_m = full["old_med"]
    d1_m = full["d1_med"]
    sec_m = full["sec_med"]
    univ_m = full["univ_med"]
    if None in (old_m, d1_m, sec_m, univ_m):
        return "樣本不足，無法比較舊榜與對照組的 10 日超額預測。"
    if old_m > d1_m and old_m > sec_m and old_m > univ_m:
        return "舊榜頭 20 的 10 日超額中位數同時高過純 1D、板塊榜、全市場，舊榜有一點橫截面排序力。"
    return "舊榜冇顯示出比當日升幅或板塊相對強度更好的 10 日預測。"


def write_report(
    daily_stats: pd.DataFrame,
    feature_days: list[dt.date],
    scored_days: list[dt.date],
    conclusion: str,
) -> None:
    full = summarize_block(daily_stats, "full")
    years = sorted(daily_stats["year"].unique()) if len(daily_stats) else []

    lines: list[str] = []
    lines.append("# Ranking predictive-power backtest report")
    lines.append("")
    lines.append("Validates the **OLD** ranking formula only (no weight optimization, no buy/sell advice).")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Feature / sample start: **{FEATURE_START.isoformat()}**")
    lines.append(f"- Price lookback download from: **{DOWNLOAD_START}** (for SMA200 / 220-day history)")
    lines.append(f"- Forward label: excess_10 vs SPY over {FWD_DAYS} ticker trading days")
    lines.append(f"- Top N per rank: **{TOP_N}** (ticker alpha tiebreak)")
    lines.append(f"- Feature files written: **{len(feature_days)}** days")
    lines.append(f"- Scored days (with T+{FWD_DAYS}): **{len(scored_days)}**")
    if feature_days:
        lines.append(f"- Feature date range: {feature_days[0]} → {feature_days[-1]}")
    if scored_days:
        lines.append(f"- Scored date range: {scored_days[0]} → {scored_days[-1]}")
    lines.append(f"- Avg names ranked per scored day: **{fmt_num(full['avg_names'], 1)}**")
    lines.append("")
    lines.append("### OLD score formula")
    lines.append("")
    lines.append("```python")
    lines.append("norm_d200 = min(max(d200, -15.0), 25.0)")
    lines.append("norm_d50  = min(max(d50,  -10.0), 20.0)")
    lines.append("norm_d20  = min(max(d20,   -8.0), 15.0)")
    lines.append("score_old = 3*daily_change + 0.8*norm_d20 + 0.3*norm_d50 + 0.1*norm_d200")
    lines.append("            - (3 if d20 >= 10 else 0)")
    lines.append("```")
    lines.append("")
    lines.append("## Full period")
    lines.append("")
    lines.append("| Rank | Median of daily top20 excess_10 medians | Avg hit rate | % days top20 excess median > 0 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| rank_old | {fmt_num(full['old_med'])} | {fmt_pct(full['old_hit'])} | {fmt_pct(full['old_pos_pct'], 1)} |"
    )
    lines.append(
        f"| rank_1d | {fmt_num(full['d1_med'])} | {fmt_pct(full['d1_hit'])} | {fmt_pct(full['d1_pos_pct'], 1)} |"
    )
    lines.append(
        f"| rank_sector | {fmt_num(full['sec_med'])} | {fmt_pct(full['sec_hit'])} | {fmt_pct(full['sec_pos_pct'], 1)} |"
    )
    lines.append(
        f"| universe (all tradable) | {fmt_num(full['univ_med'])} | — | — |"
    )
    lines.append("")
    lines.append("### Spreads (median of daily differences)")
    lines.append("")
    lines.append(f"- median(old − 1d) = **{fmt_num(full['diff_old_1d'])}**")
    lines.append(f"- median(old − sector) = **{fmt_num(full['diff_old_sec'])}**")
    lines.append(f"- median(old − universe) = **{fmt_num(full['diff_old_univ'])}**")
    lines.append("")

    if years:
        lines.append("## By year")
        lines.append("")
        for y in years:
            block = summarize_block(daily_stats[daily_stats["year"] == y], str(y))
            lines.append(f"### {y} (n={block['n_days']} days)")
            lines.append("")
            lines.append("| Rank | Med top20 excess | Avg hit | % days med>0 |")
            lines.append("|---|---:|---:|---:|")
            lines.append(
                f"| rank_old | {fmt_num(block['old_med'])} | {fmt_pct(block['old_hit'])} | {fmt_pct(block['old_pos_pct'], 1)} |"
            )
            lines.append(
                f"| rank_1d | {fmt_num(block['d1_med'])} | {fmt_pct(block['d1_hit'])} | {fmt_pct(block['d1_pos_pct'], 1)} |"
            )
            lines.append(
                f"| rank_sector | {fmt_num(block['sec_med'])} | {fmt_pct(block['sec_hit'])} | {fmt_pct(block['sec_pos_pct'], 1)} |"
            )
            lines.append(
                f"| universe | {fmt_num(block['univ_med'])} | — | — |"
            )
            lines.append("")
            lines.append(
                f"Spreads: old−1d={fmt_num(block['diff_old_1d'])}, "
                f"old−sector={fmt_num(block['diff_old_sec'])}, "
                f"old−univ={fmt_num(block['diff_old_univ'])}"
            )
            lines.append("")

    # sample days
    lines.append("## Sample days")
    lines.append("")
    if len(daily_stats):
        sample = daily_stats.head(5)
        if len(daily_stats) > 10:
            sample = pd.concat([daily_stats.head(3), daily_stats.tail(3)])
        lines.append("| date | n | old_med | 1d_med | sec_med | univ_med | old_hit |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sample.iterrows():
            lines.append(
                f"| {r['date']} | {int(r['n_names'])} | {fmt_num(r['old_excess_med'])} | "
                f"{fmt_num(r['d1_excess_med'])} | {fmt_num(r['sec_excess_med'])} | "
                f"{fmt_num(r['univ_excess_med'])} | {fmt_pct(r['old_hit'])} |"
            )
    else:
        lines.append("_No scored days._")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(conclusion)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} (Asia/Hong_Kong). "
        "Observational only; not investment advice._"
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    print("=== Ranking predictive-power backtest ===")
    print(f"Feature start: {FEATURE_START} | Download lookback from: {DOWNLOAD_START}")
    print(f"Storage: {'parquet' if USE_PARQUET else 'csv'}")

    tickers = get_sp500_tickers()
    # Ensure BRK-B style symbols and SPY
    if "BRK-B" not in tickers:
        tickers.append("BRK-B")
    if "SPY" not in tickers:
        tickers.append("SPY")
    # de-dupe preserve order
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]
    print(f"Universe size: {len(tickers)}")

    by_ticker = download_prices(tickers)
    if "SPY" not in by_ticker:
        print("ERROR: SPY data missing; cannot compute excess returns.", file=sys.stderr)
        return 1
    spy = by_ticker["SPY"]

    latest_close = max(df.index.max() for df in by_ticker.values()).date()
    print(f"Latest closed trading day in data: {latest_close}")

    dates = trading_dates_in_range(by_ticker, FEATURE_START, latest_close)
    print(f"Feature trading days from {FEATURE_START}: {len(dates)}")

    # --- Pass 1: write daily features (skip existing) ---
    feature_days: list[dt.date] = []
    for asof in dates:
        d = asof.date()
        path = feature_path(d)
        if path.exists():
            feature_days.append(d)
            continue
        print(f"Building features {d} ...")
        day_df = build_features_for_day(by_ticker, asof)
        if day_df.empty:
            print(f"  empty — skip")
            continue
        save_features(day_df, d)
        feature_days.append(d)
        print(f"  wrote {len(day_df)} names → {path.name}")

    feature_days = sorted(feature_days)
    print(f"Feature days available: {len(feature_days)}")

    # --- Pass 2: score + labels ---
    daily_rows: list[dict] = []
    scored_days: list[dt.date] = []

    for d in feature_days:
        day_df = load_features(d)
        if day_df.empty:
            continue
        asof = pd.Timestamp(d)

        # attach excess_10
        excesses = []
        for ticker in day_df["ticker"]:
            ex = forward_excess(by_ticker, ticker, asof, spy)
            excesses.append(ex)
        day_df = day_df.copy()
        day_df["excess_10"] = excesses
        labeled = day_df.dropna(subset=["excess_10"])
        if labeled.empty:
            # features-only day (inside last 10 sessions)
            continue

        labeled = labeled.copy()
        labeled["score_old"] = [
            score_old(r.daily_change, r.d20, r.d50, r.d200) for r in labeled.itertuples()
        ]
        labeled["hit"] = (labeled["excess_10"] > 0).astype(int)

        top_old = rank_top20(labeled, "score_old")
        top_1d = rank_top20(labeled, "daily_change")
        top_sec = rank_top20(labeled, "sector_pct")

        old_med = float(top_old["excess_10"].median())
        d1_med = float(top_1d["excess_10"].median())
        sec_med = float(top_sec["excess_10"].median())
        univ_med = float(labeled["excess_10"].median())

        daily_rows.append(
            {
                "date": d.isoformat(),
                "year": d.year,
                "n_names": int(len(labeled)),
                "old_excess_med": old_med,
                "d1_excess_med": d1_med,
                "sec_excess_med": sec_med,
                "univ_excess_med": univ_med,
                "old_hit": float(top_old["hit"].mean()),
                "d1_hit": float(top_1d["hit"].mean()),
                "sec_hit": float(top_sec["hit"].mean()),
                "diff_old_1d": old_med - d1_med,
                "diff_old_sec": old_med - sec_med,
                "diff_old_univ": old_med - univ_med,
            }
        )
        scored_days.append(d)

    daily_stats = pd.DataFrame(daily_rows)
    full = summarize_block(daily_stats, "full") if len(daily_stats) else {
        "old_med": None, "d1_med": None, "sec_med": None, "univ_med": None,
        "avg_names": None,
    }
    conclusion = conclusion_sentence(full if len(daily_stats) else {
        "old_med": None, "d1_med": None, "sec_med": None, "univ_med": None,
    })

    write_report(daily_stats, feature_days, scored_days, conclusion)

    print("\n=== Headline ===")
    print(f"scored_days={len(scored_days)}  avg_names={fmt_num(full.get('avg_names'), 1)}")
    print(f"old_med={fmt_num(full.get('old_med'))}  1d={fmt_num(full.get('d1_med'))}  "
          f"sector={fmt_num(full.get('sec_med'))}  univ={fmt_num(full.get('univ_med'))}")
    print(f"conclusion: {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
