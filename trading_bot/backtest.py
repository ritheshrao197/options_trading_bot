"""
Backtest engine for a weekly short-strangle strategy on a single index.

StrategyParams holds the knobs that the optimizer (optimizer.py) is
allowed to tune: target_delta, stop_loss_mult, profit_target_pct.

Deliberately NOT tunable by the optimizer: position size (LOTS). Letting
an optimizer maximize backtest returns by increasing leverage is a classic
way to get a great-looking backtest and a blown-up live account, so
position sizing stays a fixed, deliberate risk decision, not a search
variable.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from options_pricing import bs_price, find_strike_for_delta


LOT_SIZE = 75            # NIFTY lot size -- check current exchange spec, this changes over time
LOTS = 1                 # fixed on purpose -- see module docstring
RISK_FREE_RATE = 0.065
DAYS_PER_EXPIRY = 5      # simplification: treat every 5 trading days as one weekly cycle


@dataclass
class StrategyParams:
    target_delta: float = 0.18       # ~delta of each leg sold
    stop_loss_mult: float = 1.5      # exit if loss > this x premium collected
    profit_target_pct: float = 0.5   # exit if profit >= this fraction of premium collected

    def key(self):
        return (round(self.target_delta, 3), round(self.stop_loss_mult, 3),
                round(self.profit_target_pct, 3))


@dataclass
class Trade:
    week_start_day: int
    entry_spot: float
    call_strike: float
    put_strike: float
    premium_collected: float
    exit_day: int
    exit_reason: str
    pnl: float


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = None
    starting_capital: float = 0.0


def run_backtest(prices: np.ndarray, params: StrategyParams = None,
                  starting_capital: float = 200_000.0, iv_estimate: float = 0.13) -> BacktestResult:
    """
    prices: array of daily spot prices (index 0 = day 0)
    iv_estimate: flat implied vol assumption used to price the options
                 (a real system pulls live IV from the option chain instead)
    """
    params = params or StrategyParams()
    n_days = len(prices) - 1
    capital = starting_capital
    equity_curve = [capital]
    trades: List[Trade] = []

    day = 0
    while day + DAYS_PER_EXPIRY <= n_days:
        entry_spot = prices[day]
        T_entry = DAYS_PER_EXPIRY / 252.0

        call_k = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE,
                                        iv_estimate, "CE", params.target_delta)
        put_k = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE,
                                       iv_estimate, "PE", params.target_delta)

        call_entry_price = bs_price(entry_spot, call_k, T_entry, RISK_FREE_RATE, iv_estimate, "CE")
        put_entry_price = bs_price(entry_spot, put_k, T_entry, RISK_FREE_RATE, iv_estimate, "PE")
        premium_collected = (call_entry_price + put_entry_price) * LOT_SIZE * LOTS

        exit_day, exit_reason, pnl = day + DAYS_PER_EXPIRY, "expiry", None

        for d in range(day + 1, day + DAYS_PER_EXPIRY + 1):
            days_left = (day + DAYS_PER_EXPIRY) - d
            T_now = max(days_left, 0) / 252.0
            spot_now = prices[d]

            call_now = bs_price(spot_now, call_k, T_now, RISK_FREE_RATE, iv_estimate, "CE")
            put_now = bs_price(spot_now, put_k, T_now, RISK_FREE_RATE, iv_estimate, "PE")
            current_value = (call_now + put_now) * LOT_SIZE * LOTS
            mtm_pnl = premium_collected - current_value

            if mtm_pnl <= -params.stop_loss_mult * premium_collected:
                exit_day, exit_reason, pnl = d, "stop_loss", mtm_pnl
                break
            if mtm_pnl >= params.profit_target_pct * premium_collected:
                exit_day, exit_reason, pnl = d, "profit_target", mtm_pnl
                break

        if pnl is None:
            final_spot = prices[day + DAYS_PER_EXPIRY]
            call_final = bs_price(final_spot, call_k, 0, RISK_FREE_RATE, iv_estimate, "CE")
            put_final = bs_price(final_spot, put_k, 0, RISK_FREE_RATE, iv_estimate, "PE")
            pnl = premium_collected - (call_final + put_final) * LOT_SIZE * LOTS

        capital += pnl
        equity_curve.append(capital)
        trades.append(Trade(day, entry_spot, call_k, put_k, premium_collected,
                             exit_day, exit_reason, pnl))

        day += DAYS_PER_EXPIRY

    return BacktestResult(trades=trades, equity_curve=np.array(equity_curve),
                           starting_capital=starting_capital)
