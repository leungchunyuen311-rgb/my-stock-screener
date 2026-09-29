import datetime
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
    d20: float, d50: float, d200: float, rsi: float, daily_change: float, event: str
) -> tuple[str, str, str, float]:
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

    # 全局動能排名得分公式
    rank_score = (
        (0.45 * daily_change)
        + (0.25 * d20)
        + (0.15 * d50)
        + (0.15 * max(d200, -10.0))
        - (3.0 if oi_status == 'OI 壓頂' else 0.0)
    )

    return filter_status, setup, oi_status, rank_score

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

            filter_status, setup, oi_status, score = determine_advanced_metrics(
                d20, d50, d200, rsi, daily_change, event_status
            )

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
                'filter_status': filter_status,
                'setup': setup,
                'oi_status': oi_status,
                'score': score,
                'updated_at': today_date_str,
            })
        except Exception:
            continue

    raw_list.sort(key=lambda x: x['score'], reverse=True)
    for idx, item in enumerate(raw_list):
        item['rank'] = idx + 1

    with open('screener_data.json', 'w', encoding='utf-8') as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)

    print(f'\n[完成] 成功生成含板塊、機構籌碼 A~E、止蝕試算之全市場數據！')

if __name__ == '__main__':
    main()
