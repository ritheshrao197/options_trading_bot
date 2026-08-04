"""
Broker interface for Backtest -> Paper Trading -> Live Trading transition.

- PaperBroker: Simulated execution engine that fetches live market quotes (Kite Connect, Angel One, or yfinance fallback)
  and tracks simulated orders, positions, and MTM PnL under real market timing with ZERO financial risk.
- AngelOneBroker: 100% FREE Angel One SmartAPI broker wrapper for live order execution.
- KiteBroker: Zerodha Kite Connect broker wrapper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from options_pricing import bs_price, find_strike_for_delta
from backtest import LOT_SIZE, LOTS, RISK_FREE_RATE, DAYS_PER_EXPIRY


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    transaction_type: str
    quantity: int
    filled_price: float
    status: str = "COMPLETE"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class BrokerInterface(ABC):
    @abstractmethod
    def get_ltp(self, symbol: str) -> float:
        """Fetch Last Traded Price (LTP) for symbol."""
        ...

    @abstractmethod
    def place_order(self, symbol: str, qty: int, transaction_type: str,
                     order_type: str = "MARKET", price: float = None) -> OrderResult:
        """transaction_type: 'BUY' or 'SELL'"""
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Returns dict of symbol -> net quantity."""
        ...


class PaperBroker(BrokerInterface):
    """
    Paper Trading Broker with real live quote streaming capabilities.
    Connects to live quote feeds (yfinance / Angel One / KiteConnect). Fills orders
    instantly at last quoted price with zero financial risk.
    """

    def __init__(self, kite_instance=None, angel_instance=None, default_source: str = "yfinance"):
        self.kite = kite_instance
        self.angel = angel_instance
        self.default_source = default_source
        self._positions: Dict[str, int] = {}
        self._order_history: List[OrderResult] = []
        self._order_counter: int = 0
        self._manual_ltps: Dict[str, float] = {}

    def set_manual_ltp(self, symbol: str, price: float):
        """Override or inject manual price for testing."""
        self._manual_ltps[symbol] = float(price)

    def get_ltp(self, symbol: str) -> float:
        """
        Fetches live LTP.
        1. Checks manual test price if set.
        2. Tries Angel One / KiteConnect if connected.
        3. Uses yfinance live quote endpoint as free fallback.
        """
        if symbol in self._manual_ltps:
            return self._manual_ltps[symbol]

        # 1. Try Angel One SmartAPI if available (100% FREE)
        if self.angel is not None:
            try:
                return self.angel.get_ltp(symbol)
            except Exception as e:
                print(f"[PaperBroker] Angel One LTP fetch failed for {symbol}: {e}")

        # 2. Try KiteConnect if available
        if self.kite is not None:
            try:
                quote = self.kite.ltp([symbol])
                if symbol in quote and "last_price" in quote[symbol]:
                    return float(quote[symbol]["last_price"])
            except Exception as e:
                print(f"[PaperBroker] KiteConnect LTP fetch failed for {symbol}: {e}")

        # 3. Try yfinance fallback for stock / index symbols (100% FREE)
        try:
            import yfinance as yf
            yf_symbol = "^NSEI" if "NIFTY" in symbol.upper() and not symbol.startswith("^") else symbol
            ticker = yf.Ticker(yf_symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price is not None:
                return float(fast_info.last_price)

            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty and "Close" in hist.columns:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"[PaperBroker] yfinance LTP fetch failed for {symbol}: {e}")

        # Fallback default price if market is closed / offline
        return 24500.0

    def place_order(self, symbol: str, qty: int, transaction_type: str,
                     order_type: str = "MARKET", price: float = None) -> OrderResult:
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:04d}"

        fill_price = price if price is not None else self.get_ltp(symbol)
        signed_qty = qty if transaction_type.upper() == "BUY" else -qty

        current_qty = self._positions.get(symbol, 0)
        new_qty = current_qty + signed_qty

        if new_qty == 0:
            del self._positions[symbol]
        else:
            self._positions[symbol] = new_qty

        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type.upper(),
            quantity=qty,
            filled_price=round(float(fill_price), 2),
            status="COMPLETE",
        )
        self._order_history.append(result)
        print(f"[PAPER BROKER] Executed {transaction_type.upper()} {qty}x {symbol} @ Rs. {fill_price:.2f} (Order ID: {order_id})")
        return result

    def get_positions(self) -> Dict[str, int]:
        return dict(self._positions)

    def get_orders(self) -> List[OrderResult]:
        return list(self._order_history)


class AngelOneBroker(BrokerInterface):
    """
    100% FREE Broker API Integration using Angel One SmartAPI.
    No monthly developer fees. Supports live quotes, historical data,
    and automated order placement.
    """

    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: str):
        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError as e:
            raise ImportError("Please install smartapi-python and pyotp: pip install smartapi-python pyotp") from e

        self.api_key = api_key
        self.client_id = client_id
        self.smart_api = SmartConnect(api_key=api_key)

        totp = pyotp.TOTP(totp_secret).now()
        data = self.smart_api.generateSession(client_id, password, totp)
        if not data.get("status"):
            raise ConnectionError(f"Angel One SmartAPI login failed: {data.get('message')}")

        self.jwt_token = data["data"]["jwtToken"]
        self.refresh_token = data["data"]["refreshToken"]

    def get_ltp(self, symbol: str, exchange: str = "NSE", symbol_token: str = "99926000") -> float:
        """Fetch live LTP from Angel One SmartAPI."""
        res = self.smart_api.ltpData(exchange, symbol, symbol_token)
        if res.get("status") and "data" in res and "ltp" in res["data"]:
            return float(res["data"]["ltp"])

        try:
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            return float(ticker.fast_info.last_price)
        except Exception:
            return 24500.0

    def place_order(self, symbol: str, qty: int, transaction_type: str,
                     order_type: str = "MARKET", price: float = None,
                     exchange: str = "NFO", symbol_token: str = "") -> OrderResult:
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": symbol_token,
            "transactiontype": transaction_type.upper(),
            "exchange": exchange,
            "ordertype": order_type.upper(),
            "producttype": "CARRYFORWARD",
            "duration": "DAY",
            "price": str(price or 0),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(qty)
        }
        res = self.smart_api.placeOrder(order_params)
        order_id = res if isinstance(res, str) else str(res.get("data", {}).get("orderid", "ANGEL-ORDER"))
        print(f"[ANGEL ONE LIVE ORDER] {transaction_type} {qty} {symbol} -> Order ID: {order_id}")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type.upper(),
            quantity=qty,
            filled_price=price or 0.0,
            status="PLACED"
        )

    def get_positions(self) -> Dict[str, int]:
        res = self.smart_api.position()
        net_positions = {}
        if res.get("status") and "data" in res:
            for pos in res["data"]:
                net_positions[pos["tradingsymbol"]] = int(pos["netqty"])
        return net_positions


class KiteBroker(BrokerInterface):
    """
    Zerodha Kite Connect Broker Wrapper.
    """

    def __init__(self, api_key: str, access_token: str):
        try:
            from kiteconnect import KiteConnect
        except ImportError as e:
            raise ImportError("Please install kiteconnect: pip install kiteconnect") from e

        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def get_ltp(self, symbol: str) -> float:
        """Fetch live LTP from Zerodha Kite Connect."""
        quote = self.kite.ltp([symbol])
        if symbol in quote:
            return float(quote[symbol]["last_price"])
        raise ValueError(f"Symbol '{symbol}' not found in Zerodha LTP response.")

    def place_order(self, symbol: str, qty: int, transaction_type: str,
                     order_type: str = "MARKET", price: float = None) -> OrderResult:
        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=transaction_type.upper(),
            quantity=qty,
            order_type=order_type,
            product=self.kite.PRODUCT_MIS,
            price=price,
        )
        print(f"[KITE BROKER LIVE ORDER] {transaction_type} {qty} {symbol} -> Order ID: {order_id}")
        return OrderResult(
            order_id=str(order_id),
            symbol=symbol,
            transaction_type=transaction_type.upper(),
            quantity=qty,
            filled_price=price or 0.0,
            status="PLACED",
        )

    def get_positions(self) -> Dict[str, int]:
        res = self.kite.positions()
        net_positions = {}
        for pos in res.get("net", []):
            net_positions[pos["tradingsymbol"]] = pos["quantity"]
        return net_positions
