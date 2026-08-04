"""
Walk-forward optimizer: this is what makes the bot "self-improving" in an
honest sense.

Naive self-improvement (re-run a grid search on all your data and adopt
whatever scored highest) mostly just overfits to noise -- it will always
find *some* parameter combo that happened to work on the data you already
have. This module instead:

  1. Splits price history into rolling (train, test) windows.
  2. For each window, grid-searches parameters on the TRAIN slice only.
  3. Evaluates the chosen params on the TEST slice (data it didn't see).
  4. Aggregates out-of-sample (OOS) performance across all windows.
  5. Only recommends adopting a new parameter set if its average OOS
     score beats the current live params' average OOS score by more
     than `min_improvement` -- a deliberate margin so a marginal,
     probably-noise difference doesn't cause the config to churn.

This still doesn't make it a guaranteed-profitable bot -- it makes the
*parameter selection process* resistant to the most common way these
projects fool their owners (great backtest, curve-fit to history).
"""

import itertools
from dataclasses import dataclass
from typing import List
import numpy as np

from backtest import run_backtest, StrategyParams, DAYS_PER_EXPIRY
from metrics import compute_metrics

PARAM_GRID = {
    "target_delta": [0.10, 0.15, 0.18, 0.22, 0.25],
    "stop_loss_mult": [1.2, 1.5, 2.0, 2.5],
    "profit_target_pct": [0.3, 0.5, 0.7],
}

MIN_TRADES_TO_SCORE = 5         # disqualify combos with too few trades to trust
MAX_DRAWDOWN_FLOOR_PCT = -50.0  # disqualify combos with catastrophic drawdown


def _param_combos():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        yield StrategyParams(**dict(zip(keys, values)))


def _score(report: dict) -> float:
    """Higher is better. -inf disqualifies a parameter combo entirely."""
    if report.get("error"):
        return float("-inf")
    if report["num_trades"] < MIN_TRADES_TO_SCORE:
        return float("-inf")
    if report["max_drawdown_pct"] < MAX_DRAWDOWN_FLOOR_PCT:
        return float("-inf")
    sharpe = report["sharpe_annualized"]
    return sharpe if sharpe == sharpe else float("-inf")  # filters NaN


@dataclass
class WindowResult:
    train_weeks: tuple
    test_weeks: tuple
    chosen_params: dict
    in_sample_score: float
    out_of_sample_score: float
    out_of_sample_report: dict


def _make_windows(total_weeks: int, train_weeks: int, test_weeks: int, step_weeks: int):
    windows, start = [], 0
    while start + train_weeks + test_weeks <= total_weeks:
        windows.append((start, start + train_weeks, start + train_weeks + test_weeks))
        start += step_weeks
    return windows


def _slice_by_week(prices: np.ndarray, week_start: int, week_end: int) -> np.ndarray:
    d0, d1 = week_start * DAYS_PER_EXPIRY, week_end * DAYS_PER_EXPIRY
    return prices[d0: d1 + 1]


def evaluate_params_on_windows(prices: np.ndarray, params: StrategyParams,
                                windows: List[tuple], starting_capital: float,
                                iv_estimate: float) -> List[float]:
    """OOS score of a FIXED param set across each window's test slice."""
    scores = []
    for (_, train_end, test_end) in windows:
        test_prices = _slice_by_week(prices, train_end, test_end)
        result = run_backtest(test_prices, params=params,
                               starting_capital=starting_capital, iv_estimate=iv_estimate)
        scores.append(_score(compute_metrics(result)))
    return scores


def walk_forward_optimize(prices: np.ndarray, train_weeks: int = 20, test_weeks: int = 10,
                           step_weeks: int = 10, starting_capital: float = 200_000.0,
                           iv_estimate: float = 0.14) -> tuple:
    """
    Returns (best_params, avg_oos_score, window_logs).
    best_params is the param set with the best AVERAGE out-of-sample score
    across all windows -- not just the best single-window winner.
    """
    total_weeks = (len(prices) - 1) // DAYS_PER_EXPIRY
    windows = _make_windows(total_weeks, train_weeks, test_weeks, step_weeks)
    if not windows:
        return None, float("-inf"), []

    oos_scores_by_key = {}
    window_logs: List[WindowResult] = []

    for (train_start, train_end, test_end) in windows:
        train_prices = _slice_by_week(prices, train_start, train_end)
        test_prices = _slice_by_week(prices, train_end, test_end)

        best_params, best_in_sample = None, float("-inf")
        for params in _param_combos():
            result = run_backtest(train_prices, params=params,
                                   starting_capital=starting_capital, iv_estimate=iv_estimate)
            score = _score(compute_metrics(result))
            if score > best_in_sample:
                best_in_sample, best_params = score, params

        oos_result = run_backtest(test_prices, params=best_params,
                                   starting_capital=starting_capital, iv_estimate=iv_estimate)
        oos_report = compute_metrics(oos_result)
        oos_score = _score(oos_report)

        key = best_params.key()
        oos_scores_by_key.setdefault(key, []).append(oos_score)

        window_logs.append(WindowResult(
            train_weeks=(train_start, train_end), test_weeks=(train_end, test_end),
            chosen_params=dict(zip(PARAM_GRID.keys(), key)),
            in_sample_score=best_in_sample, out_of_sample_score=oos_score,
            out_of_sample_report=oos_report,
        ))

    valid = {k: [s for s in v if s != float("-inf")] for k, v in oos_scores_by_key.items()}
    valid = {k: v for k, v in valid.items() if v}
    if not valid:
        return None, float("-inf"), window_logs

    avg_oos = {k: float(np.mean(v)) for k, v in valid.items()}
    best_key = max(avg_oos, key=avg_oos.get)
    best_params = StrategyParams(**dict(zip(PARAM_GRID.keys(), best_key)))

    return best_params, avg_oos[best_key], window_logs
