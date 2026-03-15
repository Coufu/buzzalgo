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
