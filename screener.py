from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaPulse · 美股量化動能篩選系統 (S&P 500 Screener PRO)
Repo: leungchunyuen311-rgb/my-stock-screener
"""

import datetime
from collections import defaultdict
import json
import os
import sys
from typing import Dict, List, Tuple, Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# 核心巨型權重
MEGA_BLUE_CHIPS = {
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSM', 'LLY', 'AVGO', 'UNH',
    'CAT', 'BRK-B', 'JNJ', 'JPM'
}

# 自訂熱門主題分類映射表（優先匹配）
SECTOR_MAP = {
    'AI & 半導體': {
        'NVDA', 'AMD', 'TSM', 'QCOM', 'ARM', 'MU', 'MRVL', 'AMAT', 'LRCX',
        'KLAC', 'INTC', 'AVGO', 'TXN', 'ADI', 'MPWR', 'ON', 'MCHP'
    },
    '雲端 & 軟件': {
        'MSFT', 'GOOGL', 'CRM', 'NOW', 'PLTR', 'SHOP', 'ADBE', 'INTU',
        'WDAY', 'SNOW', 'DDOG', 'NET', 'ZS'
    },
    '網絡安全': {'CRWD', 'PANW', 'FTNT', 'OKTA', 'CYBR'},
    '巨型權重': {
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B'
    },
    '生物醫藥': {
        'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR',
        'AMGN', 'BMY', 'GILD', 'MRNA', 'VRTX'
    },
    '金融與周期': {
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'HOOD', 'AXP', 'CAT',
        'DE', 'GE', 'BA', 'UNP'
    }
}

# 標普 500 官方 GICS 11 大板塊中文化對照
GICS_CN = {
    'Information Technology': '科技資訊',
    'Health Care': '醫療保健',
    'Financials': '金融銀行',
    'Consumer Discretionary': '可選消費',
    'Communication Services': '通訊媒體',
    'Industrials': '工業製造',
    'Consumer Staples': '民生消費',
    'Energy': '能源石油',
    'Utilities': '公用事業',
    'Real Estate': '房地產',
    'Materials': '基礎原材料',
}

GICS_MAP = {}

CHIP_ORDER = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


def get_sector(ticker: str) -> str:
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return GICS_MAP.get(ticker, '綜合 / 其他')


def get_sp500_tickers() -> list[str]:
    global GICS_MAP
    try:
        print('正在下載 S&P 500 最新成分股名單...')
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        df = pd.read_csv(url)
        for _, row in df.iterrows():
            sym = str(row['Symbol']).replace('.', '-')
            gics = str(row.get('GICS Sector', '綜合 / 其他'))
            GICS_MAP[sym] = GICS_CN.get(gics, gics)
        return [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception as e:
        print(f'下載名單失敗，使用核心備用名單: {e}')
        fallback_tickers = [
            'QCOM', 'ARM', 'AMAT', 'LRCX', 'MRVL', 'AMD', 'MSFT', 'AAPL',
            'MU', 'INTC', 'META', 'SHOP', 'LLY', 'TSM', 'PLTR', 'GOOGL',
            'CRWD', 'PANW', 'NOW', 'HOOD', 'KLAC', 'TSLA', 'AMZN', 'AVGO',
            'CRM', 'CAT', 'DELL', 'MRNA', 'WBD', 'UNH', 'NVDA', 'WDC'
        ]
        return fallback_tickers


def extract_ticker_df(data: pd.DataFrame, ticker: str, total_tickers: int) -> pd.DataFrame | None:
    """兼顧 yfinance 所有版本之 MultiIndex 欄位抽取 (Field, Ticker) 與 (Ticker, Field)"""
    if data is None or data.empty:
        return None
    try:
        if total_tickers == 1:
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna(subset=['Close']) if 'Close' in df.columns else None

        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)
            if ticker in level0:
                df = data[ticker].copy()
            elif ticker in level1:
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            df = data.copy()

        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
            return df if len(df) >= 20 else None
    except Exception:
        return None
    return None


def compute_rsi(series: pd.Series, period: int = 14) -> float:
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
    """計算歐奈爾 IBD 機構籌碼吸籌/派發評級 (A~E)
    結合收市價變動與日內實體位置 (CLV)，避免高開低走大陰燭誤計為買盤
    """
    try:
        recent = df.tail(20)
        if len(recent) < 10 or 'Volume' not in recent.columns:
            return 'C'
        
        price_diff = recent['Close'].diff()
        highs = recent['High'] if 'High' in recent.columns else recent['Close']
        lows = recent['Low'] if 'Low' in recent.columns else recent['Close']
        closes = recent['Close']
        
        denom = (highs - lows).replace(0, 0.001)
        clv = (closes - lows) / denom
        
        up_mask = (price_diff > 0) & (clv >= 0.35)
        down_mask = (price_diff < 0) | ((price_diff >= 0) & (clv < 0.25))

        up_vol = recent.loc[up_mask, 'Volume'].sum()
        down_vol = recent.loc[down_mask, 'Volume'].sum()
        ratio = (up_vol / down_vol) if down_vol > 0 else 2.0

        if ratio >= 1.35:
            return 'A'  # 強烈吸籌
        elif ratio >= 1.12:
            return 'B'  # 溫和吸籌
        elif ratio >= 0.88:
            return 'C'  # 中性持平
        elif ratio >= 0.70:
            return 'D'  # 散戶接盤
        else:
            return 'E'  # 主力大出貨
    except Exception:
        return 'C'


def classify_position(d20: float, rsi: float) -> str:
    if d20 >= 10.0 or rsi >= 74.0:
        return '過熱'
    elif 2.0 <= d20 < 10.0 and rsi >= 52.0:
        return '偏強'
    elif -2.0 <= d20 < 2.0:
        return '中性'
    elif -10.0 < d20 < -2.0 or (30.0 <= rsi < 45.0):
        return '偏弱'
    else:
        return '超賣'


def classify_category(ticker: str, d50: float, d200: float, event: str = 'Clear') -> str:
    if event == 'Block' or d200 < -5.0:
        return '留神'
    if ticker in MEGA_BLUE_CHIPS and d200 >= -4.0:
        return '穩陣'
    if d200 >= 10.0 and d50 >= 0.0:
        return '穩陣'
    if d200 >= 0.0:
        return '睇位'
    return '留神'


def determine_advanced_metrics(
    d20: float,
    d50: float,
    d200: float,
    rsi: float,
    daily_change: float,
    event: str,
    acc_dist: str,
    position: str,
    vol_ratio: float,
    sector_pct: float,
    dollar_vol_20: float,
) -> tuple[str, str, str, float]:
    """計算進階衍生狀態與新版量化排榜得分：
    回傳 (filter_status, setup, oi_status, score)
    """
    oi_status = 'OI 壓頂' if d20 >= 10.0 else ('OI 同向' if d20 <= 2.0 else '中性')

    if event == 'Block' or d200 < -15.0:
        filter_status = '規避'
    elif d20 >= 10.0 or rsi >= 73.0:
        filter_status = '不宜'
    elif d200 >= 5.0 and (-3.5 <= d20 <= 3.0) and event == 'Clear':
        filter_status = '確認'
    else:
        filter_status = '觀望'

    if d200 >= 5.0 and (-4.0 <= d20 <= 4.0) and event == 'Clear':
        setup = 'W Wheel'
    elif d200 >= 15.0 and (-3.0 <= d20 <= 4.0) and event == 'Clear':
        setup = 'L LEAPS'
    elif d200 >= 10.0 and (-3.5 <= d20 <= 2.5) and event == 'Clear':
        setup = 'P PMCC'
    elif d200 >= 20.0 and (1.0 <= d20 <= 6.0) and event == 'Clear':
        setup = 'Z ZEBRA'
    else:
        setup = '—'

    # (a) 乖離率與日內動能收窄封頂
    norm_d20 = min(max(d20, -8.0), 12.0)
    norm_d50 = min(max(d50, -10.0), 18.0)
    norm_d200 = min(max(d200, -15.0), 25.0)
    pulse = min(max(daily_change, -6.0), 8.0)

    # (b) 籌碼分 (A +6 ~ E -8)
    chip = {'A': 6.0, 'B': 3.0, 'C': 0.0, 'D': -4.0, 'E': -8.0}.get(acc_dist, 0.0)

    # (c) 位置分
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

    # (d) 量能分
    volume_score = 0.0
    if vol_ratio >= 1.5 and daily_change > 0:
        volume_score += 3.0
    if vol_ratio < 0.7 and daily_change >= 4.0:
        volume_score -= 2.0

    # (e) 板塊相對強度 (-4 到 +4)
    sector_score = (sector_pct - 0.5) * 8.0

    # (f) 過熱動態懲罰
    extension_penalty = max(0.0, d20 - 8.0) * 0.6
    if rsi >= 75.0:
        extension_penalty += 2.0

    score = (
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
        score *= 0.5

    score = round(score, 2)
    return filter_status, setup, oi_status, score


# ---------------------------------------------------------------------------
# Watchlist / 形態買點過濾層 (修正 R:R 與突破目標空間)
# ---------------------------------------------------------------------------

def compute_spy_regime(spy_close: pd.Series) -> dict:
    price = float(spy_close.iloc[-1])
    sma20 = float(spy_close.rolling(20).mean().iloc[-1])
    sma200 = (
        float(spy_close.rolling(200).mean().iloc[-1])
        if len(spy_close) >= 200
        else float(spy_close.mean())
    )
    d20 = round(((price - sma20) / sma20) * 100, 1) if sma20 else 0.0
    d200 = round(((price - sma200) / sma200) * 100, 1) if sma200 else 0.0

    if price > sma200 and d20 > -3:
        regime, label = open, 開
    elif price > sma200:
        regime, label = half, 半倉
    else:
        regime, label = closed, 關

    return {
        ticker: SPY,
        price: round(price, 2),
        sma200: round(sma200, 2),
        d20: d20,
        d200: d200,
        above_sma200: bool(price > sma200),
        regime: regime,
        regime_label: label,
    }


def evaluate_eligibility(item: dict) -> tuple[bool, str]:
    dollar_vol_20 = float(item.get('dollar_vol_20') or 0)
    acc_dist = item.get('acc_dist') or 'C'
    d200 = float(item.get('d200') or 0)
    d20 = float(item.get('d20') or 0)
    rsi = float(item.get('rsi') or 50)
    rr_ratio = float(item.get('rr_ratio') or 0)
    daily_change = float(item.get('daily_change') or 0)

    if daily_change > 8:
        return False, '待確認'
    if dollar_vol_20 < 20_000_000:
        return False, ''
    if acc_dist == 'E':
        return False, ''
    if d200 < -4:
        return False, ''
    if not (d20 < 12 and rsi < 75):
        return False, ''
    if rr_ratio < 1.8:
        return False, ''
    return True, ''


def compute_setup_tags(item: dict) -> str:
    d20 = float(item.get('d20') or 0)
    d50 = float(item.get('d50') or 0)
    d200 = float(item.get('d200') or 0)
    acc_dist = item.get('acc_dist') or 'C'
    position = item.get('position') or ''
    vol_ratio = float(item.get('vol_ratio') or 1.0)

    tags = []
    # 1. 放量突破
    if d20 >= 3.0 and vol_ratio >= 1.2 and acc_dist in ('A', 'B'):
        tags.append('放量突破')
    # 2. 回踩
    if (-2 <= d20 <= 3) and d50 >= 0 and d200 >= 0 and acc_dist in ('A', 'B'):
        tags.append('回踩')
    # 3. 收斂待破
    if abs(d20) <= 3 and d200 >= 5 and vol_ratio < 1.1:
        tags.append('收斂待破')
    # 4. 超賣吸籌
    if position == '超賣' and acc_dist in ('A', 'B') and d200 >= -4:
        tags.append('超賣吸籌')

    if not tags:
        return '僅過閘'
    return '·'.join(tags)


def watch_sort_key(item: dict) -> tuple:
    tag = item.get('setup_tag') or ''
    has_priority = 0 if any(k in tag for k in ('回踩', '收斂待破', '放量突破')) else 1
    chip = CHIP_ORDER.get(item.get('acc_dist', 'C'), 9)
    rr = -float(item.get('rr_ratio') or 0)
    sector = -float(item.get('sector_pct') or 0)
    ticker = item.get('ticker') or ''
    return (has_priority, chip, rr, sector, ticker)


def apply_watchlist_layer(raw_list: list[dict], market: dict) -> None:
    """完全因應每日市場數據動態評定推薦名單：
    純客觀量化條件：
    1. 通過個股硬閘 (eligible == True)
    2. 具備明確買點形態 (setup_tag != '僅過閘')
    3. 機構籌碼為主力吸籌 (acc_dist in ('A', 'B'))
    4. 綜合動能評分 > 0 (score > 0)
    符合條件者全數入選，按籌碼優先度與盈虧比排序，絕不人工寫死固定隻數！
    """
    recommended_items = []
    for item in raw_list:
        ok, fail_tag = evaluate_eligibility(item)
        item['eligible'] = ok
        if ok:
            item['setup_tag'] = compute_setup_tags(item)
            is_rec = (
                item['setup_tag'] != '僅過閘'
                and item.get('acc_dist') in ('A', 'B')
                and float(item.get('score', 0)) > 0
            )
            item['is_recommended'] = is_rec
            if is_rec:
                recommended_items.append(item)
            else:
                item['watch_rank'] = None
        else:
            item['setup_tag'] = fail_tag
            item['is_recommended'] = False
            item['watch_rank'] = None

    recommended_items.sort(key=watch_sort_key)
    for idx, item in enumerate(recommended_items, 1):
        item['watch_rank'] = idx


def fetch_spy_market() -> dict:
    print('正在下載 SPY 大市閘數據...')
    if yf is None:
        raise RuntimeError('yfinance not installed')
    spy_data = yf.download(
        tickers='SPY',
        period='1y',
        interval='1d',
        threads=False,
        auto_adjust=True,
        progress=False,
    )
    spy_df = extract_ticker_df(spy_data, 'SPY', 1)
    if spy_df is None or len(spy_df) < 25:
        raise RuntimeError('SPY data too short for SMA200/D20')
    return compute_spy_regime(spy_df['Close'])


def main():
    if yf is None or pd is None:
        print("未安裝 yfinance / pandas，請在具備網路與套件之環境執行。")
        return

    tickers = get_sp500_tickers()
    print(f'正在分批下載 {len(tickers)} 隻股票數據 (每批 100 隻)...')

    # 分批下載，避免 500 隻單次呼叫 timeout 或觸發限流
    chunk_size = 100
    all_raw_dfs = {}

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f'  正在下載批次 {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} ({len(chunk)} 隻)...')
        try:
            chunk_data = yf.download(
                tickers=chunk,
                period='1y',
                interval='1d',
                threads=True,
                auto_adjust=True,
                progress=False,
            )
            for t in chunk:
                df_t = extract_ticker_df(chunk_data, t, len(chunk))
                if df_t is not None and len(df_t) >= 20:
                    all_raw_dfs[t] = df_t
        except Exception as e:
            print(f'  批次異常 ({e})，逐隻嘗試備用下載...')
            for t in chunk:
                try:
                    single_data = yf.download(
                        tickers=t,
                        period='1y',
                        interval='1d',
                        auto_adjust=True,
                        progress=False,
                    )
                    df_t = extract_ticker_df(single_data, t, 1)
                    if df_t is not None and len(df_t) >= 20:
                        all_raw_dfs[t] = df_t
                except Exception:
                    continue

    print(f'成功下載並解析 {len(all_raw_dfs)} / {len(tickers)} 隻股票數據！')

    raw_list = []
    today_date_str = datetime.date.today().strftime('%Y-%m-%d')

    for ticker, df in all_raw_dfs.items():
        try:
            close_series = df['Close']
            current_price = float(close_series.iloc[-1])
            prev_price = (
                float(close_series.iloc[-2])
                if len(close_series) >= 2
                else current_price
            )

            daily_change = round(((current_price - prev_price) / prev_price) * 100, 2)

            sma20 = float(close_series.rolling(20).mean().iloc[-1])
            sma50 = (
                float(close_series.rolling(50).mean().iloc[-1])
                if len(close_series) >= 50
                else float(close_series.mean())
            )
            sma200 = (
                float(close_series.rolling(200).mean().iloc[-1])
                if len(close_series) >= 200
                else float(close_series.mean())
            )

            d20 = round(((current_price - sma20) / sma20) * 100, 1)
            d50 = round(((current_price - sma50) / sma50) * 100, 1)
            d200 = round(((current_price - sma200) / sma200) * 100, 1)

            rsi = compute_rsi(close_series, 14)
            position = classify_position(d20, rsi)

            # 1. 歐奈爾機構籌碼吸籌評級
            acc_dist = compute_acc_dist(df)

            # 2. 板塊歸屬
            sector = get_sector(ticker)

            # 3. 支撐位、阻力位、止蝕價與【修正後動態盈虧比】
            low_20d = (
                float(df['Low'].tail(20).min())
                if 'Low' in df.columns
                else current_price * 0.95
            )
            high_20d = (
                float(df['High'].tail(20).max())
                if 'High' in df.columns
                else current_price * 1.05
            )
            supports = [
                s for s in [sma20, sma50, low_20d] if s < current_price * 0.999
            ]
            support_val = round(
                max(supports) if supports else current_price * 0.96, 2
            )
            stop_loss_val = round(support_val * 0.98, 2)

            # 核心修正：突破/創高股動態向上拓展阻力空間，避免突破股 R:R 被鎖死在 0.5
            swing_range = max(high_20d - low_20d, current_price * 0.08)
            if current_price >= high_20d * 0.98:
                resistance_val = round(current_price + swing_range, 2)
            else:
                resistance_val = round(max(high_20d, current_price + swing_range * 0.5), 2)

            risk = max(current_price - stop_loss_val, 0.01)
            reward = max(resistance_val - current_price, 0.01)
            rr_ratio = round(reward / risk, 1)

            event_status = 'Clear'
            event_date = ''
            category = classify_category(ticker, d50, d200, event_status)

            iv_val = round(35.0 + abs(d20) * 1.4, 1)
            iv_level = (
                '極平'
                if iv_val < 25
                else (
                    '偏平'
                    if iv_val < 38
                    else (
                        '中性'
                        if iv_val < 55
                        else ('偏貴' if iv_val < 70 else '極貴')
                    )
                )
            )

            # 成交量與 20 日均值
            vol_ratio = 1.0
            dollar_vol_20 = 0.0
            try:
                if 'Volume' in df.columns:
                    vol_series = df['Volume'].dropna()
                    if len(vol_series) >= 2:
                        today_vol = float(vol_series.iloc[-1])
                        prior = vol_series.iloc[:-1].tail(20)
                        mean_prior = float(prior.mean()) if len(prior) > 0 else 0.0
                        if mean_prior > 0 and today_vol == today_vol:
                            vol_ratio = today_vol / mean_prior
                        else:
                            vol_ratio = 1.0
                    dv = (df['Close'] * df['Volume']).dropna().tail(20)
                    if len(dv) > 0:
                        dollar_vol_20 = float(dv.mean())
            except Exception:
                vol_ratio = 1.0
                dollar_vol_20 = 0.0

            raw_list.append({
                'ticker': ticker,
                'name': ticker,
                'price': round(current_price, 2),
                'daily_change': daily_change,
                'category': category,
                'position': position,
                'sector': sector,
                'acc_dist': acc_dist,
                'support': support_val,
                'stop_loss': stop_loss_val,
                'resistance': resistance_val,
                'rr_ratio': rr_ratio,
                'd20': d20,
                'd50': d50,
                'd200': d200,
                'rsi': rsi,
                'iv': iv_val,
                'iv_level': iv_level,
                'event': event_status,
                'event_date': event_date,
                'vol_ratio': round(vol_ratio, 3),
                'dollar_vol_20': round(dollar_vol_20, 2),
                'updated_at': today_date_str,
            })
        except Exception as e:
            continue

    # 安全防護：若沒有解析到任何股票，絕不覆蓋舊文件為空！
    if len(raw_list) == 0:
        print("❌ 錯誤：未能成功解析任何個股數據，保留既有數據以防網站清空！")
        return

    # 第二輪：同板塊相對強度 sector_pct 與打分
    sector_groups: dict[str, list[dict]] = defaultdict(list)
    for item in raw_list:
        sector_groups[item['sector']].append(item)

    for sector, members in sector_groups.items():
        n = len(members)
        if n <= 1:
            for m in members:
                m['sector_pct'] = 0.5
        else:
            ordered = sorted(members, key=lambda x: x['daily_change'])
            min_c = ordered[0]['daily_change']
            max_c = ordered[-1]['daily_change']
            if min_c == max_c:
                for m in members:
                    m['sector_pct'] = 0.5
            else:
                for i, m in enumerate(ordered):
                    m['sector_pct'] = round(i / (n - 1), 4)

    for item in raw_list:
        filter_status, setup, oi_status, score = determine_advanced_metrics(
            item['d20'],
            item['d50'],
            item['d200'],
            item['rsi'],
            item['daily_change'],
            item['event'],
            item['acc_dist'],
            item['position'],
            item['vol_ratio'],
            item['sector_pct'],
            item['dollar_vol_20'],
        )
        item['filter_status'] = filter_status
        item['setup'] = setup
        item['oi_status'] = oi_status
        item['score'] = score

    raw_list.sort(key=lambda x: x['score'], reverse=True)
    for idx, item in enumerate(raw_list):
        item['rank'] = idx + 1

    # 第三輪：大市閘與精選觀察層
    try:
        market = fetch_spy_market()
    except Exception as e:
        print(f'警告: SPY 大市閘下載失敗，預設半倉: {e}')
        market = {
            'ticker': 'SPY',
            'price': None,
            'sma200': None,
            'd20': None,
            'd200': None,
            'above_sma200': None,
            'regime': 'half',
            'regime_label': '半倉',
            'max_watch': 4,
            'error': str(e),
        }

    apply_watchlist_layer(raw_list, market)

    payload = {
        'updated_at': today_date_str,
        'market': market,
        'stocks': raw_list,
    }

    with open('screener_data.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_elig = sum(1 for x in raw_list if x.get('eligible'))
    n_watch = sum(1 for x in raw_list if x.get('watch_rank'))
    print(
        f'\n[完成] 大市閘={market.get("regime_label")} '
        f'(max_watch={market.get("max_watch")}) | '
        f'共解析股票={len(raw_list)} 隻 | eligible={n_elig} watch={n_watch}'
    )


if __name__ == '__main__':
    main()
