#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收市入貨檢查 — print tickers with entry signals after hard gates + setups."""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# Theme pools (deduped later). Always include SPY.
THEME_POOLS = [
    # AI & 半導體
    [
        "NVDA", "AMD", "TSM", "QCOM", "ARM", "MU", "MRVL", "AMAT", "LRCX",
        "KLAC", "INTC", "AVGO", "TXN", "ADI", "MPWR", "ON", "MCHP",
    ],
    # 雲端 & 軟件
    [
        "MSFT", "GOOGL", "CRM", "NOW", "PLTR", "SHOP", "ADBE", "INTU",
        "WDAY", "SNOW", "DDOG", "NET", "ZS",
    ],
    # 網絡安全
    ["CRWD", "PANW", "FTNT", "OKTA", "CYBR"],
    # 巨型權重
    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B"],
    # 生物醫藥
    [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
        "AMGN", "BMY", "GILD", "MRNA", "VRTX",
    ],
    # 金融與周期
    [
        "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "HOOD", "AXP", "CAT",
        "DE", "GE", "BA", "UNP",
    ],
]


def default_tickers() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pool in THEME_POOLS:
        for t in pool:
            if t not in seen:
                seen.add(t)
                out.append(t)
    if "SPY" not in seen:
        out.append("SPY")
    return out


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Simple 14-day mean RSI (same as screener.py — not Wilder)."""
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(float(val), 1) if not pd.isna(val) else 50.0
    except Exception:
        return 50.0


def compute_acc_dist(df: pd.DataFrame) -> str:
    """Chip A–E from 20-day up/down volume ratio with CLV check."""
    try:
        recent = df.tail(20)
        if len(recent) < 10 or "Volume" not in recent.columns:
            return "C"
        price_diff = recent["Close"].diff()
        highs = recent['High'] if 'High' in recent.columns else recent['Close']
        lows = recent['Low'] if 'Low' in recent.columns else recent['Close']
        closes = recent['Close']
        denom = (highs - lows).replace(0, 0.001)
        clv = (closes - lows) / denom

        up_mask = (price_diff > 0) & (clv >= 0.35)
        down_mask = (price_diff < 0) | ((price_diff >= 0) & (clv < 0.25))

        up_vol = recent.loc[up_mask, "Volume"].sum()
        down_vol = recent.loc[down_mask, "Volume"].sum()
        ratio = (up_vol / down_vol) if down_vol > 0 else 2.0
        if ratio >= 1.35:
            return "A"
        if ratio >= 1.12:
            return "B"
        if ratio >= 0.88:
            return "C"
        if ratio >= 0.70:
            return "D"
        return "E"
    except Exception:
        return "C"


def _single_ticker_df(data: pd.DataFrame, ticker: str, n_tickers: int) -> pd.DataFrame | None:
    try:
        if n_tickers > 1:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    return None
                df = data[ticker].copy()
            else:
                df = data.copy()
        else:
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return None


def analyze_ticker(df: pd.DataFrame, ticker: str, account: float) -> dict | None:
    """Return entry candidate dict, or None if hard gates / setup / signal fail."""
    if len(df) < 220:
        return None

    close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    day_low = float(df["Low"].iloc[-1]) if "Low" in df.columns else close
    day_vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0

    prior = df.iloc[:-1]
    prior20 = prior.tail(20)
    if len(prior20) < 20:
        return None

    high_20 = float(prior20["High"].max()) if "High" in prior20.columns else close * 1.05
    low_20 = float(prior20["Low"].min()) if "Low" in prior20.columns else close * 0.95
    avg_vol_20 = float(prior20["Volume"].mean()) if "Volume" in prior20.columns else 0.0
    dollar_vol_20 = float((prior20["Close"] * prior20["Volume"]).mean())

    close_series = df["Close"]
    sma20 = float(close_series.rolling(20).mean().iloc[-1])
    sma50 = float(close_series.rolling(50).mean().iloc[-1])
    sma200 = float(close_series.rolling(200).mean().iloc[-1])

    d20 = round(((close - sma20) / sma20) * 100, 1) if sma20 else 0.0
    d50 = round(((close - sma50) / sma50) * 100, 1) if sma50 else 0.0
    d200 = round(((close - sma200) / sma200) * 100, 1) if sma200 else 0.0
    rsi = compute_rsi(close_series, 14)
    acc_dist = compute_acc_dist(df)
    daily_change = round(((close - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
    vol_ratio = (day_vol / avg_vol_20) if avg_vol_20 > 0 else 0.0

    # Support = highest of SMA20/SMA50/20d-low that are below close
    candidates = [s for s in (sma20, sma50, low_20) if s < close * 0.999]
    if not candidates:
        return None
    support = max(candidates)
    stop = support * 0.98
    risk = max(close - stop, 0.01)

    # 核心修復：突破/破頂股上方空間動態測算
    swing_range = max(high_20 - low_20, close * 0.08)
    if close >= high_20 * 0.98:
        target = close + swing_range
    else:
        target = max(high_20, close + swing_range * 0.5)
    reward = max(target - close, close * 0.05)
    rr = reward / risk

    # --- Hard gates (all must pass) ---
    if dollar_vol_20 < 20_000_000:
        return None
    if acc_dist == "E":
        return None
    if d200 < -4:
        return None
    if not (d20 < 12 and rsi < 75):
        return None
    if rr < 1.8:  # 合理盈虧比門檻
        return None
    if daily_change > 8:
        return None

    # --- Setups (at least one) ---
    # 核心修復：收斂待破檢查前期 3 日縮量蓄勢，避免與當日突破放量衝突
    prior3_vol = float(prior.tail(3)["Volume"].mean()) if len(prior) >= 3 else avg_vol_20
    is_contracting = prior3_vol <= avg_vol_20 * 1.15

    setups: list[str] = []
    if (-2 <= d20 <= 3) and d50 >= 0 and d200 >= 0 and acc_dist in ("A", "B"):
        setups.append("回踩")
    if abs(d20) <= 3 and d200 >= 5 and (vol_ratio < 1.1 or is_contracting):
        setups.append("收斂待破")
    if not setups:
        return None

    # --- Entry signals (at least one) ---
    signals: list[str] = []
    if close >= high_20 * 0.99 and vol_ratio >= 1.2:
        signals.append("突破")
    if "回踩" in setups and close >= support and day_low <= support * 1.01 and vol_ratio >= 1.1:
        signals.append("守支撐")
    if not signals:
        return None

    shares = int(account * 0.005 / risk)
    if shares < 1:
        return None

    return {
        "ticker": ticker,
        "signals": signals,
        "setups": setups,
        "close": round(close, 2),
        "stop": round(stop, 2),
        "rr": round(rr, 1),
        "shares": shares,
        "has_huicai": "回踩" in setups,
    }


def market_regime(spy_df: pd.DataFrame) -> str:
    close = float(spy_df["Close"].iloc[-1])
    sma20 = float(spy_df["Close"].rolling(20).mean().iloc[-1])
    sma200 = float(spy_df["Close"].rolling(200).mean().iloc[-1])
    d20 = ((close - sma20) / sma20) * 100 if sma20 else 0.0

    if close <= sma200:
        return "關"
    if d20 <= -3:
        return "半倉"
    return "開"


def format_line(item: dict) -> str:
    signal = "·".join(item["signals"])
    setup = "·".join(item["setups"])
    return (
        f"{item['ticker']:<6} {signal} {setup:<8} "
        f"收市 {item['close']:.2f}  止蝕 {item['stop']:.2f}  "
        f"R:R {item['rr']:.1f}  建議股數 {item['shares']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="收市入貨檢查")
    parser.add_argument("--account", type=float, default=100000, help="帳戶資金 (預設 100000)")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="指定股票代號；唔傳就用主題池",
    )
    parser.add_argument("--force", action="store_true", help="大市關時依然強制列出防守名單")
    args = parser.parse_args()

    if args.tickers:
        tickers = list(dict.fromkeys(args.tickers))
        if "SPY" not in tickers:
            tickers.append("SPY")
    else:
        tickers = default_tickers()

    if yf is None or pd is None:
        print("未安裝 yfinance / pandas。讀取 screener_data.json 作離線驗證...", file=sys.stderr)
        data_path = "screener_data.json"
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            market = payload.get("market", {})
            print(f"大市: {market.get('regime_label', '—')} (SPY ${market.get('price', 0)})")
            stocks = payload.get("stocks", [])
            watch_list = [s for s in stocks if s.get("watch_rank") is not None]
            watch_list.sort(key=lambda x: x["watch_rank"])
            if not watch_list:
                print("今日沒有同時過閘同埋有收市訊號嘅股票。")
                return 0
            for s in watch_list:
                risk = max(s['price'] - s['stop_loss'], 0.01)
                shares = int(args.account * 0.005 / risk)
                print(
                    f"{s['ticker']:<6} 守支撐 {s.get('setup_tag', '回踩'):<8} "
                    f"收市 {s['price']:.2f}  止蝕 {s['stop_loss']:.2f}  "
                    f"R:R {s['rr_ratio']:.1f}  建議股數 {shares}"
                )
            return 0
        else:
            print("未找到 screener_data.json。請在聯網環境執行。", file=sys.stderr)
            return 1

    print(f"下載 {len(tickers)} 隻日線 (1y / 1d)...", file=sys.stderr)
    data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        threads=True,
        auto_adjust=True,
        progress=False,
    )

    n = len(tickers)
    spy_df = _single_ticker_df(data, "SPY", n)
    if spy_df is None or len(spy_df) < 220:
        print("大市: 關", flush=True)
        print("SPY 數據不足，無法判斷大市閘。", file=sys.stderr)
        return 1

    label = market_regime(spy_df)
    print(f"大市: {label}", flush=True)
    if label == "關" and not args.force:
        print("⚠️ 大市處於 200MA 年線下方（大市閘：關），依系統風控指引停止開新倉。如需檢視逆市形態請加 --force。")
        return 0

    results: list[dict] = []
    for t in tickers:
        if t == "SPY":
            continue
        df = _single_ticker_df(data, t, n)
        if df is None:
            continue
        item = analyze_ticker(df, t, args.account)
        if item is not None:
            results.append(item)

    results.sort(key=lambda x: (0 if x["has_huicai"] else 1, -x["rr"]))
    # 完全因應每日數據動態輸出全部符合條件的股票
    # if args.limit: results = results[:args.limit]

    if not results:
        print("今日沒有同時過閘同埋有收市訊號嘅股票。")
        return 0

    for item in results:
        print(format_line(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
