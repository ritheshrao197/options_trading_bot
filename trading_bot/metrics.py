"""Performance metrics computed from a BacktestResult."""

import numpy as np
from backtest import BacktestResult


def compute_metrics(result: BacktestResult, trades_per_year: int = 52) -> dict:
    trades = result.trades
    equity = result.equity_curve

    if not trades:
        return {"error": "No trades were generated -- check date range / parameters.",
                "num_trades": 0, "sharpe_annualized": float("nan"),
                "max_drawdown_pct": 0.0}

    pnls = np.array([t.pnl for t in trades])
    total_return_pct = (equity[-1] - equity[0]) / equity[0] * 100

    years = len(trades) / trades_per_year
    cagr_pct = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else float("nan")

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown_pct = drawdowns.min() * 100

    win_rate_pct = (pnls > 0).mean() * 100
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0.0
    avg_loss = pnls[pnls <= 0].mean() if (pnls <= 0).any() else 0.0

    per_trade_returns = pnls / result.starting_capital
    sharpe = (per_trade_returns.mean() / per_trade_returns.std() * np.sqrt(trades_per_year)
              if per_trade_returns.std() > 0 else float("nan"))

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "num_trades": len(trades),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate_pct": round(win_rate_pct, 2),
        "avg_win_rs": round(avg_win, 2),
        "avg_loss_rs": round(avg_loss, 2),
        "sharpe_annualized": round(sharpe, 2) if sharpe == sharpe else float("nan"),
        "exit_reason_breakdown": exit_reasons,
        "final_capital_rs": round(equity[-1], 2),
    }
