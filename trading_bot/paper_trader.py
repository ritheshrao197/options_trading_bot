"""
Live Paper-Trading Strategy Runner.

Executes the weekly short-strangle strategy loop against live market quotes
using PaperBroker with zero financial risk.

Tests the full code path under real market timing:
- Order sequencing (SELL CE & PE -> BUY CE & PE)
- Real-time MTM PnL tracking
- Live stop-loss & profit-target triggers
- Position tracking & audit log
"""

import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict

from backtest import StrategyParams, LOT_SIZE, LOTS, RISK_FREE_RATE, DAYS_PER_EXPIRY
from broker_interface import BrokerInterface, PaperBroker, OrderResult
from options_pricing import bs_price, find_strike_for_delta


@dataclass
class PaperPosition:
    entry_time: str
    entry_spot: float
    call_strike: float
    put_strike: float
    call_symbol: str
    put_symbol: str
    premium_collected: float
    status: str = "OPEN"
    exit_time: Optional[str] = None
    exit_spot: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None


class PaperTrader:
    def __init__(self, broker: Optional[BrokerInterface] = None, params: Optional[StrategyParams] = None,
                 starting_capital: float = 200_000.0, iv_estimate: float = 0.14, symbol: str = "^NSEI"):
        self.broker = broker or PaperBroker()
        self.params = params or StrategyParams()
        self.capital = starting_capital
        self.iv_estimate = iv_estimate
        self.symbol = symbol
        self.active_position: Optional[PaperPosition] = None
        self.trade_history: List[PaperPosition] = []

    def open_strangle(self) -> PaperPosition:
        """Fetch live LTP, calculate strikes, and place initial short strangle order pair."""
        spot_price = self.broker.get_ltp(self.symbol)
        T_entry = DAYS_PER_EXPIRY / 252.0

        call_k = find_strike_for_delta(spot_price, T_entry, RISK_FREE_RATE,
                                        self.iv_estimate, "CE", self.params.target_delta)
        put_k = find_strike_for_delta(spot_price, T_entry, RISK_FREE_RATE,
                                       self.iv_estimate, "PE", self.params.target_delta)

        call_entry_price = bs_price(spot_price, call_k, T_entry, RISK_FREE_RATE, self.iv_estimate, "CE")
        put_entry_price = bs_price(spot_price, put_k, T_entry, RISK_FREE_RATE, self.iv_estimate, "PE")
        premium_collected = (call_entry_price + put_entry_price) * LOT_SIZE * LOTS

        call_sym = f"NIFTY-CE-{int(call_k)}"
        put_sym = f"NIFTY-PE-{int(put_k)}"

        print("\n" + "=" * 60)
        print(f"[PAPER TRADER] Opening Short Strangle @ Spot: Rs. {spot_price:.2f}")
        print(f"  Live Params     : Delta={self.params.target_delta}, SL={self.params.stop_loss_mult}x, PT={int(self.params.profit_target_pct * 100)}%")
        print(f"  Call Leg (CE)   : Strike Rs. {call_k:.0f} @ Premium Rs. {call_entry_price:.2f}")
        print(f"  Put Leg (PE)    : Strike Rs. {put_k:.0f} @ Premium Rs. {put_entry_price:.2f}")
        print(f"  Total Premium   : Rs. {premium_collected:.2f}")
        print("=" * 60)

        # Place paper orders
        self.broker.place_order(call_sym, qty=LOT_SIZE * LOTS, transaction_type="SELL", price=call_entry_price)
        self.broker.place_order(put_sym, qty=LOT_SIZE * LOTS, transaction_type="SELL", price=put_entry_price)

        pos = PaperPosition(
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            entry_spot=spot_price,
            call_strike=call_k,
            put_strike=put_k,
            call_symbol=call_sym,
            put_symbol=put_sym,
            premium_collected=premium_collected,
        )
        self.active_position = pos
        return pos

    def check_position_status(self, current_spot: float, days_elapsed: float = 0.5) -> Dict:
        """Calculate live MTM PnL and evaluate stop-loss / profit-target triggers."""
        if not self.active_position or self.active_position.status != "OPEN":
            return {"status": "NO_POSITION"}

        pos = self.active_position
        days_left = max(DAYS_PER_EXPIRY - days_elapsed, 0)
        T_now = days_left / 252.0

        call_now = bs_price(current_spot, pos.call_strike, T_now, RISK_FREE_RATE, self.iv_estimate, "CE")
        put_now = bs_price(current_spot, pos.put_strike, T_now, RISK_FREE_RATE, self.iv_estimate, "PE")

        current_option_val = (call_now + put_now) * LOT_SIZE * LOTS
        mtm_pnl = pos.premium_collected - current_option_val

        sl_threshold = -self.params.stop_loss_mult * pos.premium_collected
        pt_threshold = self.params.profit_target_pct * pos.premium_collected

        triggered_reason = None
        if mtm_pnl <= sl_threshold:
            triggered_reason = "stop_loss"
        elif mtm_pnl >= pt_threshold:
            triggered_reason = "profit_target"

        return {
            "status": "OPEN",
            "current_spot": round(current_spot, 2),
            "mtm_pnl": round(mtm_pnl, 2),
            "current_option_val": round(current_option_val, 2),
            "sl_threshold": round(sl_threshold, 2),
            "pt_threshold": round(pt_threshold, 2),
            "triggered_reason": triggered_reason,
        }

    def close_strangle(self, exit_spot: float, reason: str) -> PaperPosition:
        """Close active strangle position by placing buyback orders."""
        if not self.active_position:
            raise ValueError("No active position to close.")

        pos = self.active_position
        T_now = 0.0  # exit calculation
        call_exit = bs_price(exit_spot, pos.call_strike, T_now, RISK_FREE_RATE, self.iv_estimate, "CE")
        put_exit = bs_price(exit_spot, pos.put_strike, T_now, RISK_FREE_RATE, self.iv_estimate, "PE")
        exit_val = (call_exit + put_exit) * LOT_SIZE * LOTS

        final_pnl = pos.premium_collected - exit_val

        # Execute exit paper orders
        self.broker.place_order(pos.call_symbol, qty=LOT_SIZE * LOTS, transaction_type="BUY", price=call_exit)
        self.broker.place_order(pos.put_symbol, qty=LOT_SIZE * LOTS, transaction_type="BUY", price=put_exit)

        pos.status = "CLOSED"
        pos.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pos.exit_spot = exit_spot
        pos.exit_reason = reason
        pos.pnl = round(final_pnl, 2)

        self.capital += pos.pnl
        self.trade_history.append(pos)
        self.active_position = None

        print("\n" + "=" * 60)
        print(f"[PAPER TRADER] CLOSED POSITION ({reason.upper()}) @ Spot: Rs. {exit_spot:.2f}")
        print(f"  Realized Trade PnL : Rs. {pos.pnl:+.2f}")
        print(f"  Updated Capital     : Rs. {self.capital:.2f}")
        print("=" * 60 + "\n")
        return pos

    def run_live_loop(self, poll_interval_sec: float = 2.0, max_ticks: Optional[int] = 5):
        """
        Runs the live strategy monitoring loop under real market timing.
        Polls live quotes from PaperBroker, tracks MTM PnL, and auto-triggers exits.
        """
        print(f"[PAPER TRADING LOOP] Starting live strategy runner on {self.symbol}...")
        self.open_strangle()

        tick = 0
        try:
            while max_ticks is None or tick < max_ticks:
                tick += 1
                current_spot = self.broker.get_ltp(self.symbol)
                status_info = self.check_position_status(current_spot, days_elapsed=tick * 0.1)

                if status_info["status"] == "OPEN":
                    print(f"  [Tick {tick:02d}] Live Spot: Rs. {status_info['current_spot']} | MTM PnL: Rs. {status_info['mtm_pnl']:+0.2f} (SL: Rs. {status_info['sl_threshold']} / PT: Rs. {status_info['pt_threshold']})")

                    if status_info["triggered_reason"]:
                        reason = status_info["triggered_reason"]
                        print(f"\n⚡ TRIGGERED EXIT CONDITION: {reason.upper()}!")
                        self.close_strangle(current_spot, reason)
                        break

                time.sleep(poll_interval_sec)

        except KeyboardInterrupt:
            print("\n[PaperTrader] Live loop interrupted by user.")

        if self.active_position and self.active_position.status == "OPEN":
            current_spot = self.broker.get_ltp(self.symbol)
            print("\n[PaperTrader] Session end reached. Closing active position.")
            self.close_strangle(current_spot, "session_end")

        return self.trade_history
