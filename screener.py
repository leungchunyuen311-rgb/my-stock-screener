import datetime
import io
import json
import pandas as pd
import requests
import yfinance as yf

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,'
        ' like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
}


def get_sp1500_tickers() -> list[str]:
  """抓取 S&P 1500 (S&P 500 + S&P 400 + S&P 600) 全市場股票名單"""
  all_tickers = []

  # 1. 抓取 S&P 500 (大盤股 ~503 隻)
  try:
    print('正在下載 S&P 500 名單...')
    url_500 = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    df_500 = pd.read_csv(url_500)
    all_tickers.extend(
        [str(t).replace('.', '-') for t in df_500['Symbol'].tolist()]
    )
    print(f'✓ S&P 500 取得 {len(all_tickers)} 隻')
  except Exception as e:
    print(f'S&P 500 下載異常: {e}')

  # 2. 抓取 S&P 400 MidCap (中型成長股 ~400 隻)
  try:
    print('正在下載 S&P 400 (中型股) 名單...')
    r400 = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
        headers=HEADERS,
        timeout=10,
    )
    tables = pd.read_html(io.StringIO(r400.text))
    for t in tables:
      for col in ['Symbol', 'Ticker symbol', 'Ticker']:
        if col in t.columns:
          tickers_400 = [str(x).replace('.', '-') for x in t[col].tolist()]
          all_tickers.extend(tickers_400)
          print(f'✓ S&P 400 取得 {len(tickers_400)} 隻')
          break
  except Exception as e:
    print(f'S&P 400 讀取異常: {e}')

  # 3. 抓取 S&P 600 SmallCap (小型潛力股 ~600 隻)
  try:
    print('正在下載 S&P 600 (小型股) 名單...')
    r600 = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies',
        headers=HEADERS,
        timeout=10,
    )
    tables = pd.read_html(io.StringIO(r600.text))
    for t in tables:
      for col in ['Symbol', 'Ticker symbol', 'Ticker']:
        if col in t.columns:
          tickers_600 = [str(x).replace('.', '-') for x in t[col].tolist()]
          all_tickers.extend(tickers_600)
          print(f'✓ S&P 600 取得 {len(tickers_600)} 隻')
          break
  except Exception as e:
    print(f'S&P 600 讀取異常: {e}')

  # 去除重複與無效符號
  unique_tickers = list(
      dict.fromkeys([t.strip().upper() for t in all_tickers if t and len(t) < 6])
  )
  print(f'\n🎯 全市場 S&P 1500 總計準備追蹤：{len(unique_tickers)} 隻股票！')
  return (
      unique_tickers
      if len(unique_tickers) > 100
      else ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']
  )


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


def classify_position(d20: float, rsi: float) -> str:
  if d20 >= 10.0 or rsi >= 74.0:
    return '過熱'
  elif 2.0 <= d20 < 10.0 and rsi >= 54.0:
    return '偏強'
  elif -2.0 <= d20 < 2.0:
    return '中性'
  elif -10.0 < d20 < -2.0 or (30.0 <= rsi < 45.0):
    return '偏弱'
  else:
    return '超賣'


def determine_equity_strategy(
    d20: float, d50: float, d200: float, rsi: float, rvol: float
) -> tuple[str, str, str]:
  """正股專屬策略引擎：判定 (建議 Advice, 形態 Setup, 理由)"""
  # 1. 深度破位熊市
  if d200 < -15.0:
    return '避開', '—', '處於長期下跌熊市，反彈多為逃命波'

  # 2. 短線過熱提示減持 (嚴禁追高)
  if d20 >= 10.0 or rsi >= 74.0:
    return (
        '減持',
        '高位整固',
        f'短線偏離均線過遠 (D20 +{d20}%, RSI {rsi})，切忌追高，宜分批止賺',
    )

  # 3. 正股「回踩買入」：大牛股回調至 20MA
  if d200 >= 6.0 and (-3.5 <= d20 <= 2.5) and (38.0 <= rsi <= 56.0):
    return (
        '買入',
        '回踩 20MA',
        f'長線強勢牛市 (D200 +{d200}%)，回踩 20MA 均線支撐，低吸性價比極高',
    )

  # 4. 正股「50MA 機構支撐」
  if d200 >= 10.0 and (-3.0 <= d50 <= 2.0):
    return (
        '買入',
        '回踩 50MA',
        f'回踩 50 天機構生命線 (D50 {d50}%)，獲中線主力買盤護盤',
    )

  # 5. 正股「放量動能突破」
  if (
      d200 >= 15.0
      and (2.0 <= d20 <= 8.0)
      and (55.0 <= rsi <= 70.0)
      and rvol >= 1.3
  ):
    return (
        '買入',
        '動態突破',
        f'主力放量推進 (平日 {rvol} 倍量，D20 +{d20}%)，多頭加速',
    )

  # 6. 正股「超賣反彈」
  if (rsi <= 32.0 or d20 <= -8.0) and d200 >= -8.0:
    return (
        '買入',
        '超賣反彈',
        f'短期急跌嚴重超賣 (RSI {rsi})，具備均值回歸強修復潛力',
    )

  # 7. 主升浪持股
  if d200 >= 10.0 and d20 > 2.0 and rsi < 70.0:
    return '持有', '穩步主升', '多頭排列穩定推進中，已持倉者可安心持有'

  return '觀望', '—', '目前處於區間震盪，等待更清晰的進場形態'


def main():
  tickers = get_sp1500_tickers()

  # 分批下載價格數據 (每批 300 隻)，避免被 Yahoo 限流
  chunk_size = 300
  results = []
  today_date_str = datetime.date.today().strftime('%Y-%m-%d')

  print(f'正在分批批次下載 {len(tickers)} 隻股票的一年日 K 線數據...')

  for i in range(0, len(tickers), chunk_size):
    chunk = tickers[i : i + chunk_size]
    print(f'處理進度: [{i+1} ~ {min(i+chunk_size, len(tickers))}] 隻...')

    try:
      data = yf.download(
          tickers=chunk,
          period='1y',
          interval='1d',
          group_by='ticker',
          threads=True,
          auto_adjust=True,
          progress=False,
      )

      for ticker in chunk:
        try:
          df = data[ticker] if len(chunk) > 1 else data
          df = df.dropna(subset=['Close'])
          if len(df) < 30:
            continue

          close_series = df['Close']
          current_price = float(close_series.iloc[-1])

          # 【防雷過濾 1】：剔除低於 $5 的垃圾仙股！
          if current_price < 5.0:
            continue

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

          high_52w = float(close_series.max())
          off_high = (
              round(((current_price - high_52w) / high_52w) * 100, 1)
              if high_52w > 0
              else 0.0
          )

          rsi = compute_rsi(close_series, 14)
          position = classify_position(d20, rsi)
          category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

          rvol = 1.0
          if 'Volume' in df.columns and len(df['Volume'].dropna()) >= 20:
            vol = float(df['Volume'].iloc[-1])
            vol20 = float(df['Volume'].rolling(20).mean().iloc[-1])
            rvol = round(vol / vol20, 2) if vol20 > 0 else 1.0

          advice, setup, reason = determine_equity_strategy(
              d20, d50, d200, rsi, rvol
          )

          results.append({
              'ticker': ticker,
              'name': ticker,
              'price': round(current_price, 2),
              'category': category,
              'position': position,
              'd20': d20,
              'd50': d50,
              'd200': d200,
              'off_high': off_high,
              'rsi': rsi,
              'rvol': rvol,
              'advice': advice,
              'setup': setup,
              'reason': reason,
              'event': 'Clear',
              'updated_at': today_date_str,
          })

        except Exception:
          continue

    except Exception as e:
      print(f'批次下載失敗: {e}')
      continue

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

  print(
      f'\n[大功告成] 成功輸出 S&P 1500 全市場共 {len(results)}'
      ' 隻高質量正股至 screener_data.json！'
  )


if __name__ == '__main__':
  main()
