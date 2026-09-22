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


def generate_buy_signal(
    d20: float, d200: float, iv: float, event_status: str
) -> tuple[str, int, str]:
  """量化買入訊號判定引擎

  返回：(訊號名稱, 推薦星級 1-5, 推薦理由)
  """
  # 1. 強勢回踩買入 (Pullback Buy - 適合正股或買 Call)
  # 條件：長線牛市 (D200>8%)、短線剛好回調到 20MA 均線支撐附近 (-3.5% 至 2.5%)、無業績地雷
  if d200 >= 8.0 and (-3.5 <= d20 <= 2.5) and event_status == 'Clear':
    score = 5 if (d200 >= 15.0 and -1.5 <= d20 <= 1.5) else 4
    return (
        '回踩買入',
        score,
        f'長線強勢 (D200 +{d200}%)，回踩 20MA 支撐位 (D20 {d20}%)，無財報風險',
    )

  # 2. 期權賣方收租 (High IV Sell Put - 適合沽 Put / Wheel)
  # 條件：中長線穩健 (D200>=0%)、IV 衝高 (>=45% 或偏貴)、回調中 (D20<=0%)、無業績地雷
  if d200 >= 0.0 and iv >= 45.0 and d20 <= 0.0 and event_status == 'Clear':
    score = 5 if (iv >= 55.0 and d20 <= -2.0) else 4
    return (
        '沽 Put',
        score,
        f'IV 偏高 ({iv}%) 且當前無財報，股價回調中，適合賺取豐厚期權權利金',
    )

  # 3. 跌深反彈 (Oversold Bounce - 短線搶反彈)
  # 條件：短線嚴重超賣偏離 (D20<=-8%)、長線未破滅 (D200>=-8%)、無業績地雷
  if d20 <= -8.0 and d200 >= -8.0 and event_status == 'Clear':
    return '超賣反彈', 4, f'短線急跌嚴重超賣 (D20 {d20}%)，具備均值回歸修復空間'

  # 4. 動量突破 (Momentum Breakout - 追趨勢)
  # 條件：極強牛市加速 (D200>=20% 且 D20>=6%)、無業績地雷
  if d200 >= 20.0 and d20 >= 6.0 and event_status == 'Clear':
    return '動量突破', 4, f'多頭強烈加速推進 (D200 +{d200}%, D20 +{d20}%)'

  return '觀望', 0, '未達特定策略標準，保持觀察'


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
      f'均線計算完成，共 {len(base_metrics)} 隻有效股票。啟動 10 線程抓取期權與財報...'
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

  # 合併數據並生成買入訊號
  final_results = []
  for ticker, base in base_metrics.items():
    det = detailed_info.get(ticker, {})
    d20 = base['d20']
    d200 = base['d200']
    iv = det.get('iv', 0.0)
    event_status = det.get('event', 'Clear')

    signal, score, reason = generate_buy_signal(d20, d200, iv, event_status)

    final_results.append({
        'ticker': ticker,
        'name': det.get('name', ticker),
        'price': base['price'],
        'category': base['category'],
        'position': base['position'],
        'd20': d20,
        'd200': d200,
        'iv': iv,
        'iv_level': det.get('iv_level', '中性'),
        'event': event_status,
        'event_date': det.get('event_date', ''),
        'signal': signal,
        'signal_score': score,
        'signal_reason': reason,
    })

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

  print(
      f'\n[完成] 成功輸出 {len(final_results)} 隻股票及其買入訊號至'
      ' screener_data.json！'
  )


if __name__ == '__main__':
  main()
