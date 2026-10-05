import datetime
from collections import defaultdict
import json
import pandas as pd
import yfinance as yf

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
    # 優先符合特色熱門主題
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    # 否則自動落入美股 11 大官方板塊，保證 500 隻股票 100% 都有分類
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
        return [
            'QCOM', 'ARM', 'AMAT', 'LRCX', 'MRVL', 'AMD', 'MSFT', 'AAPL',
            'MU', 'INTC', 'META', 'SHOP', 'LLY', 'TSM', 'PLTR', 'GOOGL',
            'CRWD', 'PANW', 'NOW', 'HOOD', 'KLAC', 'TSLA', 'AMZN', 'AVGO',
            'CRM', 'CAT', 'DELL', 'MRNA', 'WBD', 'UNH', 'NVDA', 'WDC'
        ]

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
    基於 20 日上漲日成交量 vs 下跌日成交量比例 (Up/Down Volume Ratio)
    """
    try:
        recent = df.tail(20)
        if len(recent) < 10 or 'Volume' not in recent.columns:
            return 'C'
        price_diff = recent['Close'].diff()
        up_vol = recent.loc[price_diff > 0, 'Volume'].sum()
        down_vol = recent.loc[price_diff < 0, 'Volume'].sum()
        ratio = (up_vol / down_vol) if down_vol > 0 else 2.0

        if ratio >= 1.35:
            return 'A'  # 強烈吸籌
        elif ratio >= 1.12:
            return 'B'  # 適度吸籌
        elif ratio >= 0.88:
            return 'C'  # 中性持平
        elif ratio >= 0.70:
            return 'D'  # 適度派發
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
    """Existing online score formula — DO NOT retune this round (kept for 對照)."""
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

    # Cap pulse / norm_d* to avoid permanent top dominance from extreme moves
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

    sector_score = (sector_pct - 0.5) * 8.0  # -4 to +4

    # Extension penalty scales with D20 overheat (no fixed -3 for OI 壓頂)
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
# Watchlist / hard-gate layer (product UX). Does not change score formula.
# ---------------------------------------------------------------------------

def compute_spy_regime(spy_close: pd.Series) -> dict:
    """SPY market gate from close vs SMA200 and D20.

    Returns top-level market block for screener_data.json:
      regime: 'open' | 'half' | 'closed'
      regime_label: '開' | '半倉' | '關'
      max_watch: 8 (open), 4 (half), 8 (closed — list may still show; UI marks 關)
    """
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
        regime, label, max_watch = 'open', '開', 8
    elif price > sma200:  # d20 <= -3
        regime, label, max_watch = 'half', '半倉', 4
    else:
        # Closed: new positions 0, but observation list may still show (cap 8).
        regime, label, max_watch = 'closed', '關', 8

    return {
        'ticker': 'SPY',
        'price': round(price, 2),
        'sma200': round(sma200, 2),
        'd20': d20,
        'd200': d200,
        'above_sma200': bool(price > sma200),
        'regime': regime,
        'regime_label': label,
        'max_watch': max_watch,
    }


def evaluate_eligibility(item: dict) -> tuple[bool, str]:
    """Hard gates — all must pass. Returns (eligible, setup_tag hint for failures).

    daily_change > 8 → not eligible, tagged 待確認 (event day).
    D grade is NOT excluded here (cannot rank high later via chip sort).
    """
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
    if rr_ratio < 2.0:
        return False, ''
    return True, ''


def compute_setup_tags(item: dict) -> str:
    """Morphology tags after gates pass. Multiple allowed, joined by '·'.
    If none match → 僅過閘.
    """
    d20 = float(item.get('d20') or 0)
    d50 = float(item.get('d50') or 0)
    d200 = float(item.get('d200') or 0)
    acc_dist = item.get('acc_dist') or 'C'
    position = item.get('position') or ''
    vol_ratio = float(item.get('vol_ratio') or 1.0)

    tags = []
    # 回踩
    if (-2 <= d20 <= 3) and d50 >= 0 and d200 >= 0 and acc_dist in ('A', 'B'):
        tags.append('回踩')
    # 收斂待破
    if abs(d20) <= 3 and d200 >= 5 and vol_ratio < 1:
        tags.append('收斂待破')
    # 超賣吸籌
    if position == '超賣' and acc_dist in ('A', 'B') and d200 >= -4:
        tags.append('超賣吸籌')

    if not tags:
        return '僅過閘'
    return '·'.join(tags)


def watch_sort_key(item: dict) -> tuple:
    """Sort eligible only:
    1) 回踩 or 收斂待破 first
    2) chip A>B>C>D
    3) rr_ratio high first
    4) sector_pct high first
    5) ticker alpha
    """
    tag = item.get('setup_tag') or ''
    has_priority = 0 if ('回踩' in tag or '收斂待破' in tag) else 1
    chip = CHIP_ORDER.get(item.get('acc_dist', 'C'), 9)
    rr = -float(item.get('rr_ratio') or 0)
    sector = -float(item.get('sector_pct') or 0)
    ticker = item.get('ticker') or ''
    return (has_priority, chip, rr, sector, ticker)


def apply_watchlist_layer(raw_list: list[dict], market: dict) -> None:
    """Mutates stocks: sets eligible, setup_tag, watch_rank.

    watch_rank: only top N among ordered eligible (N = market['max_watch']).
    Other eligible keep eligible=true but watch_rank=None.
    Non-eligible: watch_rank=None; setup_tag='' or '待確認'.
    """
    max_watch = int(market.get('max_watch') or 0)

    eligible_items = []
    for item in raw_list:
        ok, fail_tag = evaluate_eligibility(item)
        item['eligible'] = ok
        if ok:
            item['setup_tag'] = compute_setup_tags(item)
            eligible_items.append(item)
        else:
            item['setup_tag'] = fail_tag  # '' or '待確認'
            item['watch_rank'] = None

    eligible_items.sort(key=watch_sort_key)
    for idx, item in enumerate(eligible_items):
        if idx < max_watch:
            item['watch_rank'] = idx + 1
        else:
            item['watch_rank'] = None


def fetch_spy_market() -> dict:
    """Download SPY 1y daily and compute regime block."""
    print('正在下載 SPY 大市閘數據...')
    spy = yf.download(
        tickers='SPY',
        period='1y',
        interval='1d',
        threads=False,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = spy['Close']
    close = close.dropna()
    if len(close) < 25:
        raise RuntimeError('SPY data too short for SMA200/D20')
    return compute_spy_regime(close)


def main():
    tickers = get_sp500_tickers()
    print(f'正在批次下載 {len(tickers)} 隻股票數據...')

    data = yf.download(
        tickers=tickers,
        period='1y',
        interval='1d',
        group_by='ticker',
        threads=True,
        auto_adjust=True,
    )

    raw_list = []
    today_date_str = datetime.date.today().strftime('%Y-%m-%d')

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
            df = df.dropna(subset=['Close'])
            if len(df) < 25:
                continue

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

            # 1. 歐奈爾機構籌碼吸籌評級 (A/B/C/D/E)
            acc_dist = compute_acc_dist(df)

            # 2. 板塊歸屬
            sector = get_sector(ticker)

            # 3. 支撐位、阻力位、止蝕價與盈虧比試算
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
            resistance_val = round(max(high_20d, current_price * 1.05), 2)
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

            # vol_ratio: today Volume / mean of prior up to 20 days (exclude today)
            vol_ratio = 1.0
            dollar_vol_20 = 0.0
            try:
                if 'Volume' in df.columns:
                    vol_series = df['Volume'].dropna()
                    if len(vol_series) >= 2:
                        today_vol = float(vol_series.iloc[-1])
                        prior = vol_series.iloc[:-1].tail(20)
                        mean_prior = float(prior.mean()) if len(prior) > 0 else 0.0
                        if mean_prior > 0 and today_vol == today_vol:  # not NaN
                            vol_ratio = today_vol / mean_prior
                        else:
                            vol_ratio = 1.0
                    # dollar_vol_20: mean of Close*Volume over past 20 days
                    dv = (df['Close'] * df['Volume']).dropna().tail(20)
                    if len(dv) > 0:
                        dollar_vol_20 = float(dv.mean())
            except Exception:
                vol_ratio = 1.0
                dollar_vol_20 = 0.0

            # Pass 1: store metrics only (no score / rank yet)
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
        except Exception:
            continue

    # Pass 2: sector relative strength, then score / rank (unchanged formula)
    sector_groups: dict[str, list[dict]] = defaultdict(list)
    for item in raw_list:
        sector_groups[item['sector']].append(item)

    for sector, members in sector_groups.items():
        n = len(members)
        if n == 1:
            members[0]['sector_pct'] = 0.5
        else:
            # Percentile of daily_change within sector (0–1)
            ordered = sorted(members, key=lambda x: x['daily_change'])
            for i, m in enumerate(ordered):
                m['sector_pct'] = i / (n - 1)

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

    # Pass 3: SPY market gate + hard gates + setup tags + watch_rank
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
        f'eligible={n_elig} watch={n_watch} / {len(raw_list)}'
    )

if __name__ == '__main__':
    main()
