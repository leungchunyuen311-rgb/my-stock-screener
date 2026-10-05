# Ranking predictive-power backtest report

Validates the **OLD** ranking formula only (no weight optimization, no buy/sell advice).

## Setup

- Feature / sample start: **2026-08-01**
- Price lookback download from: **2025-01-01** (for SMA200 / 220-day history)
- Forward label: excess_10 vs SPY over 10 ticker trading days
- Top N per rank: **20** (ticker alpha tiebreak)
- Feature files written: **44** days
- Scored days (with T+10): **34**
- Feature date range: 2026-08-03 → 2026-10-02
- Scored date range: 2026-08-03 → 2026-09-18
- Avg names ranked per scored day: **501.2**

### OLD score formula

```python
norm_d200 = min(max(d200, -15.0), 25.0)
norm_d50  = min(max(d50,  -10.0), 20.0)
norm_d20  = min(max(d20,   -8.0), 15.0)
score_old = 3*daily_change + 0.8*norm_d20 + 0.3*norm_d50 + 0.1*norm_d200
            - (3 if d20 >= 10 else 0)
```

## Full period

| Rank | Median of daily top20 excess_10 medians | Avg hit rate | % days top20 excess median > 0 |
|---|---:|---:|---:|
| rank_old | -0.0084 | 45.44% | 35.3% |
| rank_1d | -0.0061 | 44.71% | 41.2% |
| rank_sector | -0.0102 | 43.53% | 29.4% |
| universe (all tradable) | -0.0173 | — | — |

### Spreads (median of daily differences)

- median(old − 1d) = **0.0001**
- median(old − sector) = **-0.0014**
- median(old − universe) = **0.0084**

## By year

### 2026 (n=34 days)

| Rank | Med top20 excess | Avg hit | % days med>0 |
|---|---:|---:|---:|
| rank_old | -0.0084 | 45.44% | 35.3% |
| rank_1d | -0.0061 | 44.71% | 41.2% |
| rank_sector | -0.0102 | 43.53% | 29.4% |
| universe | -0.0173 | — | — |

Spreads: old−1d=0.0001, old−sector=-0.0014, old−univ=0.0084

## Sample days

| date | n | old_med | 1d_med | sec_med | univ_med | old_hit |
|---|---:|---:|---:|---:|---:|---:|
| 2026-08-03 | 501 | -0.0070 | 0.0271 | -0.0004 | -0.0071 | 45.00% |
| 2026-08-04 | 501 | 0.0199 | 0.0060 | 0.0209 | 0.0006 | 70.00% |
| 2026-08-05 | 501 | -0.0162 | 0.0246 | 0.0048 | 0.0000 | 35.00% |
| 2026-09-16 | 502 | 0.0389 | 0.0283 | -0.0301 | -0.0436 | 70.00% |
| 2026-09-17 | 502 | 0.0441 | 0.0441 | -0.0083 | -0.0301 | 55.00% |
| 2026-09-18 | 502 | 0.0419 | 0.0520 | -0.0042 | -0.0305 | 60.00% |

## Conclusion

舊榜冇顯示出比當日升幅或板塊相對強度更好的 10 日預測。

---

_Generated 2026-10-05 16:58 (Asia/Hong_Kong). Observational only; not investment advice._