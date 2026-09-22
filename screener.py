from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import pandas as pd
import yfinance as yf


def get_sp500_tickers() -> list[str]:
  """自動從 Wikipedia 取得最新的 S&P 500 成分股名單"""
  try:
    print('正在從 Wikipedia 讀取 S&P 500 成分股清單...')
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    # Yahoo Finance 格式轉換 (例如 BRK.B 轉為 BRK-B)
    tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
    print(f'成功取得 {len(tickers)} 隻 S&P 500 成分股！')
    return tickers
  except Exception as e:
    print(f'讀取 Wikipedia 失敗，使用備用熱門名單: {e}')
    # 備用保底名單
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
        'QCOM',
        'MU',
        'MRVL',
        'NOW',
        'PANW',
        'UNH',
    ]


def classify_position(d20: float) -> str:
  if d20 >= 10.0:
    return '過熱 Extended'
  elif 2.0 <= d20 < 10.0:
    return '偏強'
  elif -2.0 <= d20 < 2.0:
    return '中性'
  elif -10.0 < d20 < -2.0:
    return '偏弱'
  else:
    return '超賣 Oversold'


def classify_iv_level(iv: float) -> str:
  if iv <= 0:
    return '中性'
  elif iv < 25:
    return '極平'
  elif 25 <= iv < 38:
    return '偏平'
  elif 38 <= iv < 55:
    return '中性'
  elif 55 <= iv < 70:
    return '偏貴'
  else:
    return '極貴'


def fetch_ticker_details(ticker: str, current_price: float) -> dict:
  """多線程查詢個別股票的期權 IV 及財報日"""
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

    # 2. 期權 IV (抓約 30 天到期 ATM)
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
      'iv_level': classify_iv_level(iv_val),
  }


def main():
  tickers = get_sp500_tickers()

  print(f'正在一次過下載 {len(tickers)} 隻股票的一年價格數據...')
  data = yf.download(
      tickers=tickers,
      period='1y',
      interval='1d',
      group_by='ticker',
      threads=True,
      auto_adjust=True,
  )

  base_metrics = {}
  print('正在計算技術均線 (D20, D200)...')
  for ticker in tickers:
    try:
      df = data[ticker] if len(tickers) > 1 else data
      df = df.dropna(subset=['Close'])
      if len(df) < 20:
        continue

      current_price = float(df['Close'].iloc[-1])
      sma20 = float(df['Close'].rolling(20).mean().iloc[-1])
      sma200 = (
          float(df['Close'].rolling(200).mean().iloc[-1])
          if len(df) >= 200
          else float(df['Close'].mean())
      )

      d20 = round(((current_price - sma20) / sma20) * 100, 1)
      d200 = round(((current_price - sma200) / sma200) * 100, 1)
      position = classify_position(d20)
      category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

      base_metrics[ticker] = {
          'price': round(current_price, 2),
          'd20': d20,
          'd200': d200,
          'position': position,
          'category': category,
      }
    except Exception:
      continue

  print(
      f'技術均線計算完成，共 {len(base_metrics)} 隻有效股票。啟動 10'
      ' 線程抓取期權與財報...'
  )

  # 使用 10 條線程並行查詢期權及財報
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

  # 合併數據
  final_results = []
  for ticker, base in base_metrics.items():
    det = detailed_info.get(ticker, {})
    final_results.append({
        'ticker': ticker,
        'name': det.get('name', ticker),
        'price': base['price'],
        'category': base['category'],
        'position': base['position'],
        'd20': base['d20'],
        'd200': base['d200'],
        'iv': det.get('iv', 0.0),
        'iv_level': det.get('iv_level', '中性'),
        'event': det.get('event', 'Clear'),
        'event_date': det.get('event_date', ''),
    })

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

  print(
      f'\n[大功告成] 成功輸出 screener_data.json，共 {len(final_results)} 隻股票！'
  )


if __name__ == '__main__':
  main()
