"""
Multi-Strategy Backtest Engine supporting customizable option strategies:
- Short Strangle (Weekly OTM Sell)
- Short Straddle (ATM Sell)
- Iron Condor (Defined Risk Strangle)
- Iron Butterfly (Defined Risk Straddle)
- Bull Put Spread (Credit Spread)
- Bear Call Spread (Credit Spread)
- Long Call (Directional Buy)
- Long Put (Directional Buy)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from options_pricing import bs_price, find_strike_for_delta


LOT_SIZE = 75            # NIFTY lot size
LOTS = 1
RISK_FREE_RATE = 0.065
DAYS_PER_EXPIRY = 5      # 5 trading days weekly cycle


@dataclass
class StrategyParams:
    target_delta: float = 0.18       # ~delta of main sold leg
    stop_loss_mult: float = 1.5      # exit if loss > this x net premium
    profit_target_pct: float = 0.5   # exit if profit >= this fraction of net premium

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
                  starting_capital: float = 200_000.0, iv_estimate: float = 0.13,
                  strategy_type: str = "short_strangle") -> BacktestResult:
    """
    Runs multi-strategy options backtest engine.
    strategy_type options:
    - 'short_strangle'
    - 'short_straddle'
    - 'iron_condor'
    - 'iron_butterfly'
    - 'bull_put_spread'
    - 'bear_call_spread'
    - 'long_call'
    - 'long_put'
    """
    params = params or StrategyParams()
    n_days = len(prices) - 1
    capital = starting_capital
    equity_curve = [capital]
    trades: List[Trade] = []
    st_type = strategy_type.lower()

    day = 0
    while day + DAYS_PER_EXPIRY <= n_days:
        entry_spot = prices[day]
        T_entry = DAYS_PER_EXPIRY / 252.0

        # Adjust delta for straddles vs strangles
        delta = 0.50 if "straddle" in st_type or "butterfly" in st_type else params.target_delta

        call_k = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE, iv_estimate, "CE", delta)
        put_k = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE, iv_estimate, "PE", delta)

        # Wing strikes for defined-risk strategies (Iron Condor / Butterfly / Spreads)
        call_wing = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE, iv_estimate, "CE", max(delta - 0.10, 0.05))
        put_wing = find_strike_for_delta(entry_spot, T_entry, RISK_FREE_RATE, iv_estimate, "PE", max(delta - 0.10, 0.05))

        # Entry prices
        c_price = bs_price(entry_spot, call_k, T_entry, RISK_FREE_RATE, iv_estimate, "CE")
        p_price = bs_price(entry_spot, put_k, T_entry, RISK_FREE_RATE, iv_estimate, "PE")
        c_wing_price = bs_price(entry_spot, call_wing, T_entry, RISK_FREE_RATE, iv_estimate, "CE")
        p_wing_price = bs_price(entry_spot, put_wing, T_entry, RISK_FREE_RATE, iv_estimate, "PE")

        # Premium calculation based on strategy type
        if st_type in ("short_strangle", "short_straddle"):
            net_premium = (c_price + p_price) * LOT_SIZE * LOTS
        elif st_type in ("iron_condor", "iron_butterfly"):
            net_premium = ((c_price - c_wing_price) + (p_price - p_wing_price)) * LOT_SIZE * LOTS
        elif st_type == "bull_put_spread":
            net_premium = (p_price - p_wing_price) * LOT_SIZE * LOTS
        elif st_type == "bear_call_spread":
            net_premium = (c_price - c_wing_price) * LOT_SIZE * LOTS
        elif st_type == "long_call":
            net_premium = -c_price * LOT_SIZE * LOTS
        elif st_type == "long_put":
            net_premium = -p_price * LOT_SIZE * LOTS
        else:
            net_premium = (c_price + p_price) * LOT_SIZE * LOTS

        exit_day, exit_reason, pnl = day + DAYS_PER_EXPIRY, "expiry", None

        for d in range(day + 1, day + DAYS_PER_EXPIRY + 1):
            days_left = (day + DAYS_PER_EXPIRY) - d
            T_now = max(days_left, 0) / 252.0
            spot_now = prices[d]

            c_now = bs_price(spot_now, call_k, T_now, RISK_FREE_RATE, iv_estimate, "CE")
            p_now = bs_price(spot_now, put_k, T_now, RISK_FREE_RATE, iv_estimate, "PE")
            cw_now = bs_price(spot_now, call_wing, T_now, RISK_FREE_RATE, iv_estimate, "CE")
            pw_now = bs_price(spot_now, put_wing, T_now, RISK_FREE_RATE, iv_estimate, "PE")

            if st_type in ("short_strangle", "short_straddle"):
                mtm_pnl = net_premium - (c_now + p_now) * LOT_SIZE * LOTS
            elif st_type in ("iron_condor", "iron_butterfly"):
                mtm_pnl = net_premium - ((c_now - cw_now) + (p_now - pw_now)) * LOT_SIZE * LOTS
            elif st_type == "bull_put_spread":
                mtm_pnl = net_premium - (p_now - pw_now) * LOT_SIZE * LOTS
            elif st_type == "bear_call_spread":
                mtm_pnl = net_premium - (c_now - cw_now) * LOT_SIZE * LOTS
            elif st_type == "long_call":
                mtm_pnl = (c_now * LOT_SIZE * LOTS) + net_premium
            elif st_type == "long_put":
                mtm_pnl = (p_now * LOT_SIZE * LOTS) + net_premium
            else:
                mtm_pnl = net_premium - (c_now + p_now) * LOT_SIZE * LOTS

            abs_prem = abs(net_premium) or 1000.0
            if mtm_pnl <= -params.stop_loss_mult * abs_prem:
                exit_day, exit_reason, pnl = d, "stop_loss", mtm_pnl
                break
            if mtm_pnl >= params.profit_target_pct * abs_prem:
                exit_day, exit_reason, pnl = d, "profit_target", mtm_pnl
                break

        if pnl is None:
            final_spot = prices[day + DAYS_PER_EXPIRY]
            c_end = bs_price(final_spot, call_k, 0, RISK_FREE_RATE, iv_estimate, "CE")
            p_end = bs_price(final_spot, put_k, 0, RISK_FREE_RATE, iv_estimate, "PE")
            cw_end = bs_price(final_spot, call_wing, 0, RISK_FREE_RATE, iv_estimate, "CE")
            pw_end = bs_price(final_spot, put_wing, 0, RISK_FREE_RATE, iv_estimate, "PE")

            if st_type in ("short_strangle", "short_straddle"):
                pnl = net_premium - (c_end + p_end) * LOT_SIZE * LOTS
            elif st_type in ("iron_condor", "iron_butterfly"):
                pnl = net_premium - ((c_end - cw_end) + (p_end - pw_end)) * LOT_SIZE * LOTS
            elif st_type == "bull_put_spread":
                pnl = net_premium - (p_end - pw_end) * LOT_SIZE * LOTS
            elif st_type == "bear_call_spread":
                pnl = net_premium - (c_end - cw_end) * LOT_SIZE * LOTS
            elif st_type == "long_call":
                pnl = (c_end * LOT_SIZE * LOTS) + net_premium
            elif st_type == "long_put":
                pnl = (p_end * LOT_SIZE * LOTS) + net_premium
            else:
                pnl = net_premium - (c_end + p_end) * LOT_SIZE * LOTS

        capital += pnl
        equity_curve.append(capital)
        trades.append(Trade(day, entry_spot, call_k, put_k, round(net_premium, 2),
                             exit_day, exit_reason, round(pnl, 2)))

        day += DAYS_PER_EXPIRY

    return BacktestResult(trades=trades, equity_curve=np.array(equity_curve),
                           starting_capital=starting_capital)
