"""
Strategy Tournament — Adaptive Signal Weighting
=================================================
Backtests each research strategy individually over rolling windows (30d, 60d, 90d),
computes adaptive weights based on recent performance, and writes them to strategy.json
so the live trader hot-reloads the optimal signal mix.

Usage:
    python3 tournament.py              # run full tournament, write weights
    python3 tournament.py --dry-run    # show weights without writing to strategy.json
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from backtest import simulate, BacktestResult
from research import (
    STRATEGIES,
    DEFAULT_UNIVERSE,
    fetch_research_data,
    ResearchStrategy,
)

logger = logging.getLogger("tournament")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

ET = ZoneInfo("America/New_York")
STRATEGY_JSON = Path(__file__).parent / "strategy.json"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Rolling windows to evaluate, and their weights in the combined score
WINDOWS = [
    (30, 0.5),   # 30-day window, weight 0.5 (recent performance matters most)
    (60, 0.3),   # 60-day window, weight 0.3
    (90, 0.2),   # 90-day window, weight 0.2
]

# Minimum weight — strategies below this threshold are disabled
MIN_WEIGHT = 0.1

# Map from research strategy key to the signal_type strings it produces.
# These must match the signal_type values in strategy.py's generate_signals().
STRATEGY_SIGNAL_MAP = {
    "vwap": ["vwap_reversion_long", "vwap_revert_long", "vwap_revert_short"],
    "orb": ["orb_long", "orb_short"],
    "rsi_div": ["rsi_div_bullish", "rsi_div_bearish"],
    "vol_spike": ["vol_spike_hammer", "vol_spike_star"],
    "dual_ema": ["dual_ema_long", "dual_ema_short", "dual_ema_trend"],
}

# All signal_type strings that the tournament can weight.
# Signals not listed here (e.g. higher_low_momentum, keltner_breakout, macd_crossover)
# are left unweighted (treated as weight 1.0 in strategy.py).
ALL_TOURNAMENT_SIGNALS = []
for sigs in STRATEGY_SIGNAL_MAP.values():
    ALL_TOURNAMENT_SIGNALS.extend(sigs)


def _notify_slack(text: str):
    """Send a Slack notification."""
    if not SLACK_WEBHOOK_URL:
        return
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.debug("Slack notification failed: %s", e)


def _backtest_strategy(
    strategy: ResearchStrategy,
    bars: dict[str, "pd.DataFrame"],
    window_days: int,
) -> dict:
    """Run a single strategy backtest over data trimmed to window_days.

    Returns a dict with sharpe, win_rate, profit_factor, pnl, trades.
    """
    import pandas as pd

    # Trim bars to the most recent window_days of trading data
    if not bars:
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "pnl": 0.0, "trades": 0}

    cutoff = datetime.now(ET) - __import__("datetime").timedelta(days=window_days + 5)
    trimmed = {}
    for symbol, df in bars.items():
        cutoff_ts = pd.Timestamp(cutoff)
        if df.index.tz and cutoff_ts.tz is None:
            cutoff_ts = cutoff_ts.tz_localize(df.index.tz)
        elif df.index.tz and cutoff_ts.tz is not None:
            cutoff_ts = cutoff_ts.tz_convert(df.index.tz)
        elif not df.index.tz and cutoff_ts.tz is not None:
            cutoff_ts = cutoff_ts.tz_localize(None)
        mask = df.index >= cutoff_ts
        sub = df[mask]
        if len(sub) >= 30:  # need enough bars for indicators
            trimmed[symbol] = sub

    if not trimmed:
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "pnl": 0.0, "trades": 0}

    try:
        result: BacktestResult = simulate(strategy, trimmed, portfolio_value=100_000)
        return {
            "sharpe": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "pnl": result.total_pnl,
            "trades": result.total_trades,
        }
    except Exception as e:
        logger.error("  Backtest failed: %s", e, exc_info=True)
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "pnl": 0.0, "trades": 0}


def run_tournament(
    num_symbols: int = 20,
    dry_run: bool = False,
) -> dict[str, float]:
    """Run the full tournament and return computed signal weights."""

    symbols = DEFAULT_UNIVERSE[:num_symbols]

    # Fetch data once for the longest window (90d + buffer)
    max_days = max(w for w, _ in WINDOWS)
    logger.info("Fetching %d days of data for %d symbols...", max_days, len(symbols))
    bars = fetch_research_data(symbols, max_days)
    if not bars:
        logger.error("No data fetched. Aborting tournament.")
        return {}

    logger.info("Data ready: %d symbols\n", len(bars))

    # Run each strategy over each window
    # results[strategy_key][window_days] = metrics_dict
    results: dict[str, dict[int, dict]] = {}

    for key, strategy_cls in STRATEGIES.items():
        results[key] = {}
        strategy = strategy_cls()
        logger.info("=== %s ===", strategy.name)

        for window_days, _ in WINDOWS:
            logger.info("  Window %dd ...", window_days)
            t0 = time.time()
            metrics = _backtest_strategy(strategy, bars, window_days)
            elapsed = time.time() - t0
            results[key][window_days] = metrics
            logger.info(
                "    Sharpe=%.4f  WinRate=%.1f%%  PF=%.2f  Trades=%d  P&L=$%.2f  (%.1fs)",
                metrics["sharpe"],
                metrics["win_rate"] * 100,
                metrics["profit_factor"],
                metrics["trades"],
                metrics["pnl"],
                elapsed,
            )

    # Compute combined score for each strategy
    # Score = weighted average of Sharpe across windows
    strategy_scores: dict[str, float] = {}
    for key in STRATEGIES:
        score = 0.0
        for window_days, window_weight in WINDOWS:
            sharpe = results[key].get(window_days, {}).get("sharpe", 0.0)
            score += window_weight * sharpe
        strategy_scores[key] = score
        logger.info("Combined score for %s: %.4f", key, score)

    # Zero out strategies with negative combined score
    for key in strategy_scores:
        if strategy_scores[key] < 0:
            strategy_scores[key] = 0.0

    # Normalize to produce per-signal weights
    total_score = sum(strategy_scores.values())

    signal_weights: dict[str, float] = {}
    if total_score > 0:
        for key, score in strategy_scores.items():
            weight = score / total_score
            # Apply minimum threshold
            if weight < MIN_WEIGHT:
                weight = 0.0
            for sig_type in STRATEGY_SIGNAL_MAP.get(key, []):
                signal_weights[sig_type] = round(weight, 4)
    else:
        # All strategies scored <= 0; set everything to 0
        for sig_type in ALL_TOURNAMENT_SIGNALS:
            signal_weights[sig_type] = 0.0

    # Re-normalize non-zero weights to sum to 1.0
    active_sigs = {k: v for k, v in signal_weights.items() if v > 0}
    if active_sigs:
        active_total = sum(active_sigs.values())
        for sig_type in signal_weights:
            if signal_weights[sig_type] > 0:
                signal_weights[sig_type] = round(signal_weights[sig_type] / active_total, 4)

    # --- Walk-Forward Validation ---
    # Train on first 67% of 90d data, test on last 33%
    logger.info("\n=== Walk-Forward Validation (train 67%% / test 33%%) ===")
    wf_results = {}
    for key, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls()
        train_bars = {}
        test_bars = {}
        for symbol, df in bars.items():
            split_idx = int(len(df) * 0.67)
            train_bars[symbol] = df.iloc[:split_idx]
            test_bars[symbol] = df.iloc[split_idx:]

        try:
            train_result = simulate(strategy, train_bars, portfolio_value=100_000)
            test_result = simulate(strategy, test_bars, portfolio_value=100_000)
            wf_results[key] = {
                "train_sharpe": train_result.sharpe_ratio,
                "test_sharpe": test_result.sharpe_ratio,
                "overfit_ratio": (train_result.sharpe_ratio - test_result.sharpe_ratio) / max(abs(train_result.sharpe_ratio), 0.01),
            }
            logger.info("  %s: train=%.2f test=%.2f overfit=%.1f%%",
                        key, train_result.sharpe_ratio, test_result.sharpe_ratio,
                        wf_results[key]["overfit_ratio"] * 100)
        except Exception as e:
            logger.error("  %s walk-forward failed: %s", key, e)
            wf_results[key] = {"train_sharpe": 0, "test_sharpe": 0, "overfit_ratio": 0}

    # --- Monte Carlo Simulation ---
    # Shuffle trade P&Ls 500 times to get confidence intervals
    logger.info("\n=== Monte Carlo Simulation (500 trials) ===")
    mc_results = {}
    for key, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls()
        try:
            result = simulate(strategy, bars, portfolio_value=100_000)
            if result.trades:
                pnls = [t.pnl for t in result.trades]
                mc_sharpes = []
                for _ in range(500):
                    shuffled = np.random.permutation(pnls)
                    cumulative = np.cumsum(shuffled)
                    daily_chunks = [cumulative[i:i+26] for i in range(0, len(cumulative), 26)]  # ~26 bars/day
                    if len(daily_chunks) > 1:
                        daily_returns = []
                        for chunk in daily_chunks:
                            if len(chunk) > 0:
                                daily_returns.append(chunk[-1] - (chunk[0] if len(chunk) > 1 else 0))
                        if daily_returns and np.std(daily_returns) > 0:
                            mc_sharpes.append(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
                if mc_sharpes:
                    mc_results[key] = {
                        "median_sharpe": float(np.median(mc_sharpes)),
                        "p5_sharpe": float(np.percentile(mc_sharpes, 5)),
                        "p95_sharpe": float(np.percentile(mc_sharpes, 95)),
                        "pct_positive": float(np.mean([s > 0 for s in mc_sharpes]) * 100),
                    }
                    logger.info("  %s: median=%.2f  5th=%.2f  95th=%.2f  %%positive=%.0f%%",
                                key, mc_results[key]["median_sharpe"],
                                mc_results[key]["p5_sharpe"], mc_results[key]["p95_sharpe"],
                                mc_results[key]["pct_positive"])
                else:
                    mc_results[key] = {"median_sharpe": 0, "p5_sharpe": 0, "p95_sharpe": 0, "pct_positive": 0}
            else:
                mc_results[key] = {"median_sharpe": 0, "p5_sharpe": 0, "p95_sharpe": 0, "pct_positive": 0}
        except Exception as e:
            logger.error("  %s Monte Carlo failed: %s", key, e)
            mc_results[key] = {"median_sharpe": 0, "p5_sharpe": 0, "p95_sharpe": 0, "pct_positive": 0}

    # --- Strategy Correlation ---
    # Compute daily P&L correlation between strategies
    logger.info("\n=== Strategy Correlation Matrix ===")
    daily_pnls = {}
    for key, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls()
        try:
            result = simulate(strategy, bars, portfolio_value=100_000)
            if result.trades:
                trade_pnls = {}
                for t in result.trades:
                    day = t.exit_time[:10] if t.exit_time else "unknown"
                    trade_pnls[day] = trade_pnls.get(day, 0) + t.pnl
                daily_pnls[key] = trade_pnls
        except Exception:
            pass

    correlation_matrix = {}
    strategy_keys = list(daily_pnls.keys())
    if len(strategy_keys) >= 2:
        all_days = sorted(set().union(*(set(d.keys()) for d in daily_pnls.values())))
        pnl_matrix = np.array([
            [daily_pnls[k].get(day, 0) for day in all_days]
            for k in strategy_keys
        ])
        corr = np.corrcoef(pnl_matrix)
        for i, k1 in enumerate(strategy_keys):
            for j, k2 in enumerate(strategy_keys):
                if i < j:
                    correlation_matrix[f"{k1} vs {k2}"] = round(float(corr[i, j]), 3)
                    logger.info("  %s vs %s: %.3f", k1, k2, corr[i, j])

    # Print summary
    print()
    print("=" * 60)
    print("  Tournament Results")
    print("=" * 60)
    header = f"{'Strategy':<18} {'30d Sharpe':>10} {'60d Sharpe':>10} {'90d Sharpe':>10} {'Score':>8} {'Weight':>8}"
    print(header)
    print("-" * len(header))
    for key, strategy_cls in STRATEGIES.items():
        s30 = results[key].get(30, {}).get("sharpe", 0.0)
        s60 = results[key].get(60, {}).get("sharpe", 0.0)
        s90 = results[key].get(90, {}).get("sharpe", 0.0)
        score = strategy_scores[key]
        # Weight for display: use first signal_type for this strategy
        first_sig = STRATEGY_SIGNAL_MAP.get(key, [""])[0]
        w = signal_weights.get(first_sig, 0.0)
        print(f"{key:<18} {s30:>10.4f} {s60:>10.4f} {s90:>10.4f} {score:>8.4f} {w:>7.1%}")
    print()

    print("Signal Weights:")
    for sig_type, w in sorted(signal_weights.items()):
        status = "ACTIVE" if w > 0 else "disabled"
        print(f"  {sig_type:<28} {w:.4f}  ({status})")
    print()

    # Walk-forward
    if wf_results:
        print("Walk-Forward Validation:")
        for key in STRATEGIES:
            wf = wf_results.get(key, {})
            print(f"  {key:<18} train={wf.get('train_sharpe', 0):>7.2f}  test={wf.get('test_sharpe', 0):>7.2f}  overfit={wf.get('overfit_ratio', 0):>6.0%}")
        print()

    # Monte Carlo
    if mc_results:
        print("Monte Carlo (500 trials):")
        for key in STRATEGIES:
            mc = mc_results.get(key, {})
            print(f"  {key:<18} median={mc.get('median_sharpe', 0):>7.2f}  5th={mc.get('p5_sharpe', 0):>7.2f}  95th={mc.get('p95_sharpe', 0):>7.2f}  %positive={mc.get('pct_positive', 0):>4.0f}%")
        print()

    # Correlation
    if correlation_matrix:
        print("Strategy Correlations:")
        for pair, corr in sorted(correlation_matrix.items()):
            label = "HIGH" if abs(corr) > 0.5 else "low"
            print(f"  {pair:<30} {corr:>6.3f}  ({label})")
        print()

    # Store for Slack summary
    run_tournament._wf_results = wf_results
    run_tournament._mc_results = mc_results
    run_tournament._correlation = correlation_matrix

    # Write to strategy.json
    if not dry_run:
        _write_weights(signal_weights)
        logger.info("Weights written to strategy.json")
    else:
        logger.info("DRY RUN — weights NOT written to strategy.json")

    # Slack summary
    _post_slack_summary(results, strategy_scores, signal_weights, dry_run)

    return signal_weights


def _write_weights(signal_weights: dict[str, float]):
    """Write signal_weights to strategy.json, preserving all other keys."""
    with open(STRATEGY_JSON) as f:
        config = json.load(f)

    old_weights = config.get("signal_weights", {})
    config["signal_weights"] = signal_weights
    config["signal_weights_updated_at"] = datetime.now(ET).isoformat()

    with open(STRATEGY_JSON, "w") as f:
        json.dump(config, f, indent=4)

    # Log changes
    for sig_type in sorted(set(list(signal_weights.keys()) + list(old_weights.keys()))):
        old_w = old_weights.get(sig_type, None)
        new_w = signal_weights.get(sig_type, 0.0)
        if old_w is None:
            logger.info("  NEW  %s = %.4f", sig_type, new_w)
        elif abs(old_w - new_w) > 0.001:
            logger.info("  CHG  %s: %.4f -> %.4f", sig_type, old_w, new_w)


def _post_slack_summary(
    results: dict[str, dict[int, dict]],
    strategy_scores: dict[str, float],
    signal_weights: dict[str, float],
    dry_run: bool,
):
    """Post tournament summary to Slack."""
    lines = [":trophy: *Strategy Tournament Results*"]
    if dry_run:
        lines.append("_(dry run — weights not applied)_")
    lines.append("")

    for key in STRATEGIES:
        s30 = results[key].get(30, {}).get("sharpe", 0.0)
        s60 = results[key].get(60, {}).get("sharpe", 0.0)
        s90 = results[key].get(90, {}).get("sharpe", 0.0)
        score = strategy_scores[key]
        first_sig = STRATEGY_SIGNAL_MAP.get(key, [""])[0]
        w = signal_weights.get(first_sig, 0.0)
        emoji = ":white_check_mark:" if w > 0 else ":x:"
        lines.append(
            f"{emoji} *{key}*: 30d={s30:.2f} 60d={s60:.2f} 90d={s90:.2f} | "
            f"score={score:.2f} weight={w:.1%}"
        )

    active_count = sum(1 for v in signal_weights.values() if v > 0)
    total_count = len(signal_weights)
    lines.append(f"\n{active_count}/{total_count} signal types active")

    # Walk-forward & Monte Carlo highlights
    if hasattr(run_tournament, '_wf_results'):
        wf = run_tournament._wf_results
        mc = run_tournament._mc_results
        corr = run_tournament._correlation
        lines.append("\n*Validation:*")
        for key in STRATEGIES:
            wf_data = wf.get(key, {})
            mc_data = mc.get(key, {})
            test_s = wf_data.get("test_sharpe", 0)
            pct_pos = mc_data.get("pct_positive", 0)
            if test_s != 0 or pct_pos != 0:
                lines.append(f"  {key}: walk-fwd test={test_s:.2f} | MC {pct_pos:.0f}% positive")
        if corr:
            high_corr = {k: v for k, v in corr.items() if abs(v) > 0.5}
            if high_corr:
                lines.append(f"\n:warning: High correlation: {', '.join(f'{k}={v:.2f}' for k, v in high_corr.items())}")

    _notify_slack("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Strategy Tournament — Adaptive Signal Weighting")
    parser.add_argument("--dry-run", action="store_true", help="Show weights without writing to strategy.json")
    parser.add_argument("--symbols", type=int, default=20, help="Number of symbols to test (default: 20)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Strategy Tournament")
    print(f"  {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"  Symbols: {args.symbols} | Windows: {[w for w, _ in WINDOWS]}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    weights = run_tournament(num_symbols=args.symbols, dry_run=args.dry_run)
    if not weights:
        print("Tournament failed — no weights computed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
