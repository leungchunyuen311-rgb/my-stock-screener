import datetime
import json
import pandas as pd
import yfinance as yf


def get_sp500_tickers() -> list[str]:
  """自動抓取 S&P 500 全市場完整名單 (約 503 隻)"""
  try:
    print('正在下載標普 500 最新全市場成分股名單...')
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    df = pd.read_csv(url)
    tickers = [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
    print(f'成功取得 {len(tickers)} 隻股票！')
    return tickers
  except Exception as e:
    print(f'下載名單失敗，使用備用名單: {e}')
    return [
        'NVDA',
        'AMD',
        'QCOM',
        'ARM',
        'TSM',
        'MU',
        'MRVL',
        'INTC',
        'AMAT',
        'LRCX',
        'KLAC',
        'META',
        'GOOGL',
        'AAPL',
        'MSFT',
        'AMZN',
        'AVGO',
        'CRWD',
        'PANW',
        'PLTR',
        'SHOP',
        'NOW',
        'HOOD',
        'TSLA',
        'CRM',
        'LLY',
        'CAT',
        'WDC',
        'UNH',
        'DELL',
        'MRNA',
        'WBD',
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
  elif 2.0 <= d20 < 10.0 and rsi >= 52.0:
    return '偏強'
  elif -2.0 <= d20 < 2.0:
    return '中性'
  elif -10.0 < d20 < -2.0 or (30.0 <= rsi < 45.0):
    return '偏弱'
  else:
    return '超賣'


def determine_advanced_metrics(
    d20: float,
    d50: float,
    d200: float,
    rsi: float,
    daily_change: float,
    event: str,
) -> tuple[str, str, str, float]:
  """計算 Filter, Setup, OI 及 全局排名評分 Score"""
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

  # 全局動能排名得分：當日動能爆發 (45%) + 短中長均線排列 (55%)
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
  print(f'正在批次下載 {len(tickers)} 隻股票的一年日 K 線數據...')

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
      if len(df) < 20:
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
      category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

      event_status = 'Clear'
      event_date = ''
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

  # 全市場 500 隻股票統一按評分排名，由 #1 排到 #500！
  raw_list.sort(key=lambda x: x['score'], reverse=True)
  for idx, item in enumerate(raw_list):
    item['rank'] = idx + 1

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(raw_list, f, ensure_ascii=False, indent=2)

  print(
      f'\n[完成] 成功為全市場 {len(raw_list)} 隻股票排定全局名次 (#1 ~'
      f' #{len(raw_list)})！'
  )


if __name__ == '__main__':
  main()
