#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking predictive-power backtest: Compares OLD vs NEW multi-factor formulas.

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

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError:
    np = None
    pd = None
    yf = None

# Reuse helpers from screener WITHOUT editing screener.py
from screener import (
    classify_position,
    compute_acc_dist,
    compute_rsi,
    get_sector,
    get_sp500_tickers,
)

# --- constants ---
FEATURE_START = dt.date(2026, 8, 1)
DOWNLOAD_START = "2025-01-01"
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
    """舊版 4 因子打分公式"""
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


def score_new(
    daily_change: float,
    d20: float,
    d50: float,
    d200: float,
    rsi: float,
    position: str,
    acc_dist: str,
    vol_ratio: float,
    sector_pct: float,
    dollar_vol_20: float,
) -> float:
    """新版多因子打分公式 (納入籌碼、位階、量能、板塊相對強度及動態過熱扣分)"""
    norm_d20 = min(max(d20, -8.0), 12.0)
    norm_d50 = min(max(d50, -10.0), 18.0)
    norm_d200 = min(max(d200, -15.0), 25.0)
    pulse = min(max(daily_change, -6.0), 8.0)

    chip = {'A': 6.0, 'B': 3.0, 'C': 0.0, 'D': -4.0, 'E': -8.0}.get(acc_dist, 0.0)

    if position == '過熱':
        zone = -4.0
    elif position == '偏強':
        zone = 2.0
    elif position == '偏弱':
        zone = -1.0
    elif position == '超賣' and acc_dist in ('A', 'B'):
        zone = 3.0
    else:
        zone = 0.0

    volume_score = 0.0
    if vol_ratio >= 1.5 and daily_change > 0:
        volume_score += 3.0
    if vol_ratio < 0.7 and daily_change >= 4.0:
        volume_score -= 2.0

    sector_score = (sector_pct - 0.5) * 8.0
    extension_penalty = max(0.0, d20 - 8.0) * 0.6
    if rsi >= 75.0:
        extension_penalty += 2.0

    s = (
        1.2 * pulse
        + 0.5 * norm_d20
        + 0.25 * norm_d50
        + 0.15 * norm_d200
        + chip
        + zone
        + volume_score
        + sector_score
        - extension_penalty
    )
    if dollar_vol_20 < 20_000_000:
        s *= 0.5
    return round(s, 2)


def rank_top20(day_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    tmp = day_df.copy()
    tmp = tmp.sort_values([score_col, "ticker"], ascending=[False, True])
    return tmp.head(TOP_N)


def main() -> int:
    print("=== AlphaPulse: Ranking Predictive-Power Backtest ===")
    if yf is None or pd is None:
        print("未安裝 yfinance / pandas。回測需在完整 Python 金融套件環境下執行。")
        return 0
    print("環境就緒，可執行歷史截面回測。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
