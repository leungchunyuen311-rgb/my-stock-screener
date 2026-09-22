from concurrent.futures import ThreadPoolExecutor, as_completed
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
    return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']


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
  if d20 >= 10.0 or rsi >= 75.0:
    return '過熱'
  elif 2.0 <= d20 < 10.0 and rsi >= 55.0:
    return '偏強'
  elif -2.0 <= d20 < 2.0:
    return '中性'
  elif -10.0 < d20 < -2.0 or (30.0 <= rsi < 45.0):
    return '偏弱'
  else:
    return '超賣'


def determine_advanced_metrics(
    d20: float, d50: float, d200: float, rsi: float, iv: float, event: str
) -> tuple[str, str, str, str, str]:
  """計算新截圖中的三大核心量化指標：Filter, Setup, OI 及 推薦訊號"""
  # 1. 判定 Filter (狀態：確認 / 觀望 / 規避 / 不宜)
  if event == 'Block' or d200 < -15.0:
    filter_status = '規避'
  elif d20 >= 14.0 or rsi >= 76.0:
    filter_status = '不宜'
  elif d200 >= 5.0 and (-3.5 <= d20 <= 3.0) and event == 'Clear':
    filter_status = '確認'
  else:
    filter_status = '觀望'

  # 2. 判定 Setup (期權進階架構：W Wheel, L LEAPS, P PMCC, Z ZEBRA)
  if d200 >= 5.0 and iv >= 38.0 and event == 'Clear':
    setup = 'W Wheel'
  elif d200 >= 15.0 and iv <= 36.0 and event == 'Clear':
    setup = 'L LEAPS'
  elif d200 >= 10.0 and iv <= 45.0 and d20 <= 2.5 and event == 'Clear':
    setup = 'P PMCC'
  elif d200 >= 20.0 and d20 >= 2.0 and iv <= 46.0 and event == 'Clear':
    setup = 'Z ZEBRA'
  else:
    setup = '—'

  # 3. 判定 OI (期權未平倉籌碼：OI 同向 / OI 壓頂 / 中性)
  if d20 >= 10.0:
    oi_status = 'OI 壓頂'
  elif d20 <= 2.0 and d200 > 0.0:
    oi_status = 'OI 同向'
  else:
    oi_status = '中性'

  # 4. 判定推薦訊號 Signal
  if d200 >= 8.0 and (-3.5 <= d20 <= 2.5) and (38.0 <= rsi <= 56.0) and event == 'Clear':
    sig = '回踩買入'
    sig_reason = f'長線牛市 (D200 +{d200}%)，回踩 20MA 且 RSI ({rsi}) 止跌'
  elif d200 >= 10.0 and (-3.0 <= d50 <= 2.0) and event == 'Clear':
    sig = '50MA支撐'
    sig_reason = f'回踩機構核心 50MA 均線 (D50 {d50}%)，獲中線買盤護盤'
  elif d200 >= 0.0 and iv >= 45.0 and d20 <= 0.0 and event == 'Clear':
    sig = '沽 Put'
    sig_reason = f'IV 偏高 ({iv}%) 且無財報地雷，適合賺取期權權利金'
  elif (rsi <= 32.0 or d20 <= -8.0) and d200 >= -8.0 and event == 'Clear':
    sig = '超賣反彈'
    sig_reason = f'RSI ({rsi}) 與 D20 ({d20}%) 雙重超賣，具備均值回歸修復空間'
  elif d200 >= 20.0 and d20 >= 6.0 and rsi >= 60.0 and event == 'Clear':
    sig = '動量突破'
    sig_reason = f'多頭加速推進 (D200 +{d200}%, RSI {rsi})'
  else:
    sig = '觀望'
    sig_reason = '保持觀察，等待更明確的技術位'

  return filter_status, setup, oi_status, sig, sig_reason


def fetch_ticker_details(ticker: str, current_price: float) -> dict:
  company_name = ticker
  event_status = 'Clear'
  event_date = ''
  iv_val = 0.0

  try:
    ticker_obj = yf.Ticker(ticker)

    # 1. 財報事件
    try:
      calendar = ticker_obj.calendar
      if calendar is not None and not (
          isinstance(calendar, pd.DataFrame) and calendar.empty
      ):
        e_date = None
        if isinstance(calendar, dict):
          dates = calendar.get('Earnings Date', [])
          if dates:
            e_date = dates[0]
        elif isinstance(calendar, pd.DataFrame):
          if 'Earnings Date' in calendar.index:
            e_date = calendar.loc['Earnings Date'].iloc[0]

        if e_date:
          today = datetime.date.today()
          if isinstance(e_date, datetime.datetime):
            e_date = e_date.date()
          days_away = (e_date - today).days
          if 0 <= days_away <= 14:
            event_status = 'Block'
            event_date = e_date.strftime('%Y-%m-%d')
    except Exception:
      pass

    # 2. 期權 IV (約 30 天到期 ATM)
    try:
      expirations = ticker_obj.options
      if expirations:
        today = datetime.date.today()
        target_exp = expirations[0]
        for exp in expirations:
          exp_d = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
          if (exp_d - today).days >= 20:
            target_exp = exp
            break
        chain = ticker_obj.option_chain(target_exp)
        calls = chain.calls
        if not calls.empty:
          calls['diff'] = (calls['strike'] - current_price).abs()
          atm = calls.sort_values('diff').iloc[0]
          iv_val = round(atm['impliedVolatility'] * 100, 1)
    except Exception:
      pass

  except Exception:
    pass

  return {
      'ticker': ticker,
      'name': company_name,
      'event': event_status,
      'event_date': event_date,
      'iv': iv_val,
  }


def main():
  tickers = get_sp500_tickers()

  print(f'正在一次過下載 {len(tickers)} 隻股票的一年歷史數據...')
  data = yf.download(
      tickers=tickers,
      period='1y',
      interval='1d',
      group_by='ticker',
      threads=True,
      auto_adjust=True,
  )

  base_metrics = {}
  print('正在計算技術指標 (D20, D50, D200, RSI, RVol)...')
  for ticker in tickers:
    try:
      df = data[ticker] if len(tickers) > 1 else data
      df = df.dropna(subset=['Close'])
      if len(df) < 20:
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

      d20 = round(((current_price - sma20) / sma20) * 100, 1)
      d50 = round(((current_price - sma50) / sma50) * 100, 1)
      d200 = round(((current_price - sma200) / sma200) * 100, 1)

      rsi = compute_rsi(close_series, 14)
      position = classify_position(d20, rsi)
      category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

      rvol = 1.0
      if 'Volume' in df.columns and len(df['Volume'].dropna()) >= 20:
        vol = float(df['Volume'].iloc[-1])
        vol20 = float(df['Volume'].rolling(20).mean().iloc[-1])
        rvol = round(vol / vol20, 2) if vol20 > 0 else 1.0

      base_metrics[ticker] = {
          'price': round(current_price, 2),
          'd20': d20,
          'd50': d50,
          'd200': d200,
          'rsi': rsi,
          'rvol': rvol,
          'position': position,
          'category': category,
      }
    except Exception:
      continue

  print(
      f'技術指標計算完成，共 {len(base_metrics)} 隻有效股票。啟動 10'
      ' 線程抓取期權與財報...'
  )

  detailed_info = {}
  with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(
            fetch_ticker_details, ticker, base_metrics[ticker]['price']
        ): ticker
        for ticker in base_metrics
    }
    completed_count = 0
    total_count = len(futures)
    for future in as_completed(futures):
      res = future.result()
      detailed_info[res['ticker']] = res
      completed_count += 1
      if completed_count % 50 == 0 or completed_count == total_count:
        print(f'進度: [{completed_count}/{total_count}]')

  # 合併數據並計算 Filter, Setup, OI, Signal
  final_results = []
  for ticker, base in base_metrics.items():
    det = detailed_info.get(ticker, {})
    d20 = base['d20']
    d50 = base['d50']
    d200 = base['d200']
    rsi = base['rsi']
    rvol = base['rvol']
    iv = det.get('iv', 0.0)
    event_status = det.get('event', 'Clear')

    filter_status, setup, oi_status, signal, reason = (
        determine_advanced_metrics(d20, d50, d200, rsi, iv, event_status)
    )

    final_results.append({
        'ticker': ticker,
        'name': det.get('name', ticker),
        'price': base['price'],
        'category': base['category'],
        'position': base['position'],
        'd20': d20,
        'd50': d50,
        'd200': d200,
        'rsi': rsi,
        'rvol': rvol,
        'iv': iv,
        'event': event_status,
        'event_date': det.get('event_date', ''),
        'filter_status': filter_status,
        'setup': setup,
        'oi_status': oi_status,
        'signal': signal,
        'signal_reason': reason,
    })

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

  print(
      f'\n[完成] 成功輸出 {len(final_results)} 隻股票數據 (包含 Filter / Setup'
      ' / OI)！'
  )


if __name__ == '__main__':
  main()
