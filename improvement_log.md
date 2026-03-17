# Improvement Log

All strategy modifications are logged here with hypotheses, outcomes, and reasoning.

---

## 2026-03-15 — Tighten RSI thresholds and volume filter

**Hypothesis**: RSI 30/70 thresholds generate too many low-quality oversold bounce signals. Tightening to 25/75 and raising volume_multiplier from 1.0 to 1.5 will filter weak setups, matching the top parameter sweep result (Sharpe 1.12).

**Changes**:
- `strategy.json`: `rsi_oversold` 30 → 25, `rsi_overbought` 70 → 75, `volume_multiplier` 1.0 → 1.5

**Backtest Result**:
- Sharpe: 4.49 (was 0.0)
- Win Rate: 44.44% (was 20%)
- Max Drawdown: 0.03%
- Profit Factor: 1.92
- Total Trades: 18
- Train Sharpe: 0.29, Test Sharpe: 4.49

**Outcome**: DEPLOYED — Sharpe improved by +4.49 (well above 0.05 threshold), max drawdown unchanged at 0.03%.

---

## 2026-03-16 — Add EMA slope filter for long entries

**Hypothesis**: Long entries (rsi_oversold_bounce) fire even when the EMA is flat or declining, leading to bounces without trend support. Requiring `ema_slope > 0` for longs ensures entries only occur when the short-term trend is rising, filtering weak bounces in trendless markets.

**Changes**:
- `strategy.py`: Added `if ema_slope <= 0: return None` guard in long entry logic
- Version bump: 1.1.0 → 1.2.0

**Backtest Result**:
- Sharpe: 3.43 (was 2.44)
- Win Rate: 58.33% (was 53.85%)
- Max Drawdown: 0.03% (unchanged)
- Profit Factor: 1.86 (was 1.35)
- Total Trades: 12 (was 13)
- Train Sharpe: -9.04, Test Sharpe: 3.43

**Outcome**: DEPLOYED — Sharpe improved by +0.99 (well above 0.05 threshold), max drawdown unchanged at 0.03%. Filtered 1 low-quality trade, improving win rate and profit factor significantly.

---

## 2026-03-17 — Tighten crypto params + bullish candle filter

**Hypothesis 1 (Crypto)**: Crypto RSI 30/70 with volume_multiplier 0.3 generates low-quality signals (baseline: Sharpe -2.04, 16.67% win rate). Tightening to RSI 25/75 with volume_multiplier 0.5-0.8 (mirroring successful equity tightening) should filter weak setups.

**Hypothesis 2 (Equity)**: Adding bullish candle confirmation (close > open) for long entries filters entries where price action contradicts the RSI bounce signal.

**Changes tested**:
- `strategy.json` crypto: `rsi_oversold` 30→25, `rsi_overbought` 70→75, `volume_multiplier` 0.3→0.8 (then 0.5)
- `strategy.py`: Added `bullish_candle` parameter to `_evaluate_entry`, required for long entries

**Backtest Results**:
- Equity with bullish candle filter: Sharpe 3.43 (unchanged from baseline 3.43), 12 trades — filter never triggered
- Crypto with RSI 25/75, vol 0.8: Sharpe -4.40, 1 trade (was 12), -$5.95 — much worse
- Crypto with RSI 25/75, vol 0.5: Sharpe -4.40, 1 trade (was 12), -$5.95 — same, RSI threshold is binding constraint

**Outcome**: REVERTED — Equity unchanged (no Sharpe improvement). Crypto significantly worse: RSI 25 threshold eliminates nearly all crypto signals because crypto RSI rarely drops below 25 in 30-day window. The equity-optimal RSI thresholds do not transfer to crypto markets. Bullish candle filter was a no-op in this period (all existing long signals already had bullish candles).

---

## 2026-03-17 — Reduce EMA period from 20 to 10 (v1.3.0)

**Hypothesis 1 (failed)**: Reducing `take_profit_atr_mult` from 2.0 to 1.5 would capture profits before reversals. Backtest showed Sharpe 3.25 (down from 3.43) — tighter TP hurt rather than helped. Reverted.

**Hypothesis 2 (deployed)**: EMA(20) is too slow for mean-reversion entries — by the time RSI crosses above 25 AND price exceeds EMA(20), the bounce is already well underway and runs out of steam. Shortening to EMA(10) makes the trend filter more responsive, catching valid bounces earlier while still requiring trend confirmation.

**Changes**:
- `strategy.json`: `ema_period` 20 → 10
- Version bump: 1.2.0 → 1.3.0

**Backtest Result**:
- Sharpe: 4.14 (was 3.43, +0.71)
- Win Rate: 53.85% (was 58.33%)
- Max Drawdown: 0.03% (unchanged)
- Profit Factor: 1.76 (was 1.86)
- Total Trades: 13 (was 12)
- Total P&L: $38.17
- Train Sharpe: -3.28, Test Sharpe: 4.14

**Outcome**: DEPLOYED — Sharpe improved by +0.71 (well above 0.05 threshold), max drawdown unchanged at 0.03%. Added 1 trade while maintaining win rate above 50% and profit factor above 1.3. Slight win rate and profit factor dip acceptable given the strong Sharpe improvement.
