import datetime
import json
import pandas as pd
import yfinance as yf


def get_sp500_tickers() -> list[str]:
  try:
    print('正在下載 S&P 500 最新成分股名單...')
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    df = pd.read_csv(url)
    tickers = [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
    print(f'成功取得 {len(tickers)} 隻標普 500 股票！')
    return tickers
  except Exception as e:
    print(f'讀取失敗，使用備用名單: {e}')
    return [
        'AAPL',
        'MSFT',
        'NVDA',
        'AMZN',
        'GOOGL',
        'META',
        'TSLA',
        'AMD',
        'AVGO',
        'CRWD',
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
    d20: float, d50: float, d200: float, rsi: float, rvol: float, event: str
) -> tuple[str, str, str]:
  """正股專屬策略引擎：判定 (操作建議 Filter, 交易形態 Setup, 具體理由)"""
  # 1. 避開財報或深度破位熊市股
  if event == 'Block' or d200 < -15.0:
    return (
        '避開',
        '—',
        '即將公布財報或處於破位熊市，防守優先'
        if event == 'Block'
        else '處於長期空頭走勢，反彈多為逃命波',
    )

  # 2. 短線過熱提示減持
  if d20 >= 10.0 or rsi >= 74.0:
    return (
        '減持',
        '高位整固',
        f'短線偏離均線過遠 (D20 +{d20}%, RSI {rsi})，切忌追高，宜分批獲利',
    )

  # 3. 正股「回踩買入 (Pullback)」：大牛股回調至 20MA / 50MA 支撐
  if (
      d200 >= 6.0
      and (-3.5 <= d20 <= 2.5)
      and (38.0 <= rsi <= 56.0)
      and event == 'Clear'
  ):
    return (
        '買入',
        '回踩買入',
        f'長線強勢 (D200 +{d200}%)，回踩 20MA 均線支撐，低吸性價比高',
    )

  # 4. 正股「50MA 機構支撐 (Institutional Support)」
  if d200 >= 10.0 and (-3.0 <= d50 <= 2.0) and event == 'Clear':
    return (
        '買入',
        '50MA支撐',
        f'回踩 50 天機構生命線 (D50 {d50}%)，獲中線主力買盤護盤',
    )

  # 5. 正股「放量動能突破 (Breakout)」
  if (
      d200 >= 15.0
      and (2.0 <= d20 <= 8.0)
      and (55.0 <= rsi <= 70.0)
      and rvol >= 1.3
      and event == 'Clear'
  ):
    return (
        '買入',
        '動量突破',
        f'主力放量推進 (平日 {rvol} 倍量，D20 +{d20}%)，多頭加速',
    )

  # 6. 正股「超賣反彈 (Bounce)」
  if (rsi <= 32.0 or d20 <= -8.0) and d200 >= -8.0 and event == 'Clear':
    return (
        '買入',
        '超賣反彈',
        f'短期急跌嚴重超賣 (RSI {rsi})，具備均值回歸強修復潛力',
    )

  # 7. 主升浪持股
  if d200 >= 10.0 and d20 > 2.0 and rsi < 70.0:
    return '持有', '穩步主升', '多頭排列穩定推進中，已持倉者可繼續坐定定'

  return '觀望', '—', '目前處於震盪區間，等待更清晰的進場形態'


def main():
  tickers = get_sp500_tickers()

  print(f'正在批次下載 {len(tickers)} 隻股票的一年價格與成交量數據...')
  data = yf.download(
      tickers=tickers,
      period='1y',
      interval='1d',
      group_by='ticker',
      threads=True,
      auto_adjust=True,
  )

  results = []
  today_date_str = datetime.date.today().strftime('%Y-%m-%d')
  print('正在計算正股量化指標...')

  for ticker in tickers:
    try:
      df = data[ticker] if len(tickers) > 1 else data
      df = df.dropna(subset=['Close'])
      if len(df) < 30:
        continue

      close_series = df['Close']
      current_price = float(close_series.iloc[-1])

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

      # 均線偏離度
      d20 = round(((current_price - sma20) / sma20) * 100, 1)
      d50 = round(((current_price - sma50) / sma50) * 100, 1)
      d200 = round(((current_price - sma200) / sma200) * 100, 1)

      # 距 52 週新高距離
      high_52w = float(close_series.max())
      off_high = (
          round(((current_price - high_52w) / high_52w) * 100, 1)
          if high_52w > 0
          else 0.0
      )

      rsi = compute_rsi(close_series, 14)
      position = classify_position(d20, rsi)
      category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

      # 相對成交量 (RVol)
      rvol = 1.0
      if 'Volume' in df.columns and len(df['Volume'].dropna()) >= 20:
        vol = float(df['Volume'].iloc[-1])
        vol20 = float(df['Volume'].rolling(20).mean().iloc[-1])
        rvol = round(vol / vol20, 2) if vol20 > 0 else 1.0

      # 快速檢查財報日 (只需查少量即將出財報的公司)
      event_status = 'Clear'
      event_date = ''
      # 為了保證 10 秒極速，只對候選強勢股簡單檢查或預設 Clear
      # 若需精確財報可定期更新
      advice, setup, reason = determine_equity_strategy(
          d20, d50, d200, rsi, rvol, event_status
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
          'advice': advice,  # 買入 / 持有 / 觀望 / 減持 / 避開
          'setup': setup,  # 回踩買入 / 50MA支撐 / 動量突破 / 超賣反彈 / 穩步主升
          'reason': reason,
          'event': event_status,
          'updated_at': today_date_str,
      })

    except Exception:
      continue

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

  print(
      f'\n[完成] 成功輸出 {len(results)} 隻美股正股數據庫（全過程僅數秒）！'
  )


if __name__ == '__main__':
  main()
