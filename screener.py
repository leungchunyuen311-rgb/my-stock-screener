import datetime
import json
import pandas as pd
import yfinance as yf

# 追蹤的股票清單（可隨時在此新增更多 Ticker）
TICKERS = [
    'MRVL',
    'MU',
    'AMD',
    'NOW',
    'PANW',
    'QCOM',
    'UNH',
    'NVDA',
    'AAPL',
    'MSFT',
    'TSLA',
    'AMZN',
    'GOOGL',
    'META',
    'AVGO',
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
  if iv < 25:
    return '極平'
  elif 25 <= iv < 38:
    return '偏平'
  elif 38 <= iv < 55:
    return '中性'
  elif 55 <= iv < 70:
    return '偏貴'
  else:
    return '極貴'


def get_upcoming_earnings(ticker_obj) -> tuple[str, str]:
  try:
    calendar = ticker_obj.calendar
    if calendar is not None and not (
        isinstance(calendar, pd.DataFrame) and calendar.empty
    ):
      earnings_date = None
      if isinstance(calendar, dict):
        dates = calendar.get('Earnings Date', [])
        if dates and len(dates) > 0:
          earnings_date = dates[0]
      elif isinstance(calendar, pd.DataFrame):
        if 'Earnings Date' in calendar.index:
          earnings_date = calendar.loc['Earnings Date'].iloc[0]

      if earnings_date:
        today = datetime.date.today()
        if isinstance(earnings_date, datetime.datetime):
          earnings_date = earnings_date.date()
        days_away = (earnings_date - today).days
        if 0 <= days_away <= 14:
          return 'Block', earnings_date.strftime('%Y-%m-%d')
  except Exception:
    pass
  return 'Clear', ''


def get_current_atm_iv(ticker_obj, current_price: float) -> float:
  try:
    expirations = ticker_obj.options
    if not expirations:
      return 0.0

    today = datetime.date.today()
    target_exp = expirations[0]
    for exp in expirations:
      exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
      if (exp_date - today).days >= 20:
        target_exp = exp
        break

    chain = ticker_obj.option_chain(target_exp)
    calls = chain.calls
    if not calls.empty:
      calls['diff'] = (calls['strike'] - current_price).abs()
      atm = calls.sort_values('diff').iloc[0]
      return round(atm['impliedVolatility'] * 100, 1)
  except Exception:
    pass
  return 0.0


def main():
  print('開始下載股價歷史數據...')
  data = yf.download(
      tickers=TICKERS,
      period='1y',
      interval='1d',
      group_by='ticker',
      threads=True,
      auto_adjust=True,
  )

  results = []
  for ticker in TICKERS:
    try:
      df = data[ticker] if len(TICKERS) > 1 else data
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
      ticker_obj = yf.Ticker(ticker)

      # 取得公司簡稱
      company_name = ticker_obj.info.get('shortName', ticker)
      event_status, event_date = get_upcoming_earnings(ticker_obj)
      iv_val = get_current_atm_iv(ticker_obj, current_price)
      iv_level = classify_iv_level(iv_val)

      # 分類標籤 (穩陣 / 睇位 / 留神)
      category = '穩陣' if d200 > 10 else ('留神' if d200 < -10 else '睇位')

      results.append({
          'ticker': ticker,
          'name': company_name,
          'price': round(current_price, 2),
          'category': category,
          'position': position,
          'd20': d20,
          'd200': d200,
          'iv': iv_val,
          'iv_level': iv_level,
          'event': event_status,
          'event_date': event_date,
      })
      print(f'已完成: {ticker}')
    except Exception as e:
      print(f'處理 {ticker} 錯誤: {e}')

  with open('screener_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

  print(f'成功生成 screener_data.json，共 {len(results)} 隻股票。')


if __name__ == '__main__':
  main()
