"""
REST API Backend & Static File Server for Options Trading Bot UI.
Uses Python's standard library http.server for zero external dependencies.
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

import config_store as cfg
from backtest import run_backtest, StrategyParams, DAYS_PER_EXPIRY
from data_simulator import SyntheticUnderlyingFeed, HistoricalUnderlyingFeed
from metrics import compute_metrics
from optimizer import walk_forward_optimize, evaluate_params_on_windows, _make_windows

STARTING_CAPITAL = 200_000.0
IV_ESTIMATE = 0.14
MIN_IMPROVEMENT = 0.15


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_prices(trading_days: int, seed: int = 7, source: str = "synthetic", symbol: str = "^NSEI", csv_path: str = None, start_date: str = None, end_date: str = None):
    feed = HistoricalUnderlyingFeed(
        feed_type=source,
        symbol=symbol,
        csv_path=csv_path,
        trading_days=trading_days,
        seed=seed,
        start_price=24500.0,
        start_date=start_date,
        end_date=end_date,
    )
    return feed.generate()


def trade_to_dict(t):
    return {
        "week_start_day": int(t.week_start_day),
        "entry_spot": round(float(t.entry_spot), 2),
        "call_strike": round(float(t.call_strike), 2),
        "put_strike": round(float(t.put_strike), 2),
        "premium_collected": round(float(t.premium_collected), 2),
        "exit_day": int(t.exit_day),
        "exit_reason": t.exit_reason,
        "pnl": round(float(t.pnl), 2),
    }


class BotRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve static UI files from the 'ui' directory
        ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
        super().__init__(*args, directory=ui_dir, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/config":
            self._handle_get_config()
        elif parsed.path == "/api/history":
            self._handle_get_history()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

        if parsed.path == "/api/config":
            self._handle_post_config(body)
        elif parsed.path == "/api/run":
            self._handle_post_run(body)
        elif parsed.path == "/api/optimize":
            self._handle_post_optimize(body)
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def _send_json(self, data, status=200):
        response_bytes = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _handle_get_config(self):
        state = cfg.load_state()
        params = cfg.get_live_params(state)
        self._send_json({
            "live": {
                "target_delta": params.target_delta,
                "stop_loss_mult": params.stop_loss_mult,
                "profit_target_pct": params.profit_target_pct,
            },
            "history_count": len(state.get("history", [])),
        })

    def _handle_get_history(self):
        state = cfg.load_state()
        self._send_json(state.get("history", []))

    def _handle_post_config(self, body):
        target_delta = float(body.get("target_delta", 0.18))
        stop_loss_mult = float(body.get("stop_loss_mult", 1.5))
        profit_target_pct = float(body.get("profit_target_pct", 0.5))

        new_params = StrategyParams(
            target_delta=target_delta,
            stop_loss_mult=stop_loss_mult,
            profit_target_pct=profit_target_pct,
        )
        state = cfg.load_state()
        state["live"] = {
            "target_delta": new_params.target_delta,
            "stop_loss_mult": new_params.stop_loss_mult,
            "profit_target_pct": new_params.profit_target_pct,
        }
        cfg.save_state(state)
        self._send_json({"success": True, "live": state["live"]})

    def _handle_post_run(self, body):
        days = int(body.get("days", 252))
        seed = int(body.get("seed", 7))
        capital = float(body.get("capital", STARTING_CAPITAL))
        source = str(body.get("source", "synthetic"))
        symbol = str(body.get("symbol", "^NSEI"))
        csv_path = body.get("csv_path", None)
        strategy_type = str(body.get("strategy_type", "short_strangle"))
        start_date = body.get("start_date", None)
        end_date = body.get("end_date", None)

        if "params" in body and body["params"]:
            p = body["params"]
            params = StrategyParams(
                target_delta=float(p.get("target_delta", 0.18)),
                stop_loss_mult=float(p.get("stop_loss_mult", 1.5)),
                profit_target_pct=float(p.get("profit_target_pct", 0.5)),
            )
        else:
            state = cfg.load_state()
            params = cfg.get_live_params(state)

        prices = get_prices(
            trading_days=days,
            seed=seed,
            source=source,
            symbol=symbol,
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
        )
        result = run_backtest(
            prices,
            params=params,
            starting_capital=capital,
            iv_estimate=IV_ESTIMATE,
            strategy_type=strategy_type,
        )
        report = compute_metrics(result)

        trades_list = [trade_to_dict(t) for t in result.trades]
        equity_list = [round(float(x), 2) for x in result.equity_curve]
        underlying_prices = [round(float(x), 2) for x in prices]

        self._send_json({
            "params": {
                "target_delta": params.target_delta,
                "stop_loss_mult": params.stop_loss_mult,
                "profit_target_pct": params.profit_target_pct,
            },
            "data_source": source,
            "symbol": symbol,
            "report": report,
            "equity_curve": equity_list,
            "underlying_prices": underlying_prices,
            "trades": trades_list,
        })

    def _handle_post_optimize(self, body):
        days = int(body.get("days", 756))
        seed = int(body.get("seed", 11))
        train_weeks = int(body.get("train_weeks", 20))
        test_weeks = int(body.get("test_weeks", 10))
        step_weeks = int(body.get("step_weeks", 10))
        min_imp = float(body.get("min_improvement", MIN_IMPROVEMENT))
        auto_adopt = bool(body.get("auto_adopt", True))
        source = str(body.get("source", "synthetic"))
        symbol = str(body.get("symbol", "^NSEI"))
        csv_path = body.get("csv_path", None)

        state = cfg.load_state()
        current_params = cfg.get_live_params(state)
        prices = get_prices(
            trading_days=days,
            seed=seed,
            source=source,
            symbol=symbol,
            csv_path=csv_path,
        )

        candidate, candidate_oos_score, window_logs = walk_forward_optimize(
            prices,
            train_weeks=train_weeks,
            test_weeks=test_weeks,
            step_weeks=step_weeks,
            starting_capital=STARTING_CAPITAL,
            iv_estimate=IV_ESTIMATE,
        )

        if candidate is None:
            notes = "No valid candidate found clearing guardrails."
            cfg.record_optimization_run(
                state, current_params, float("nan"), float("nan"),
                improved=False, min_improvement=min_imp, notes=notes
            )
            self._send_json({
                "success": False,
                "reason": "No parameter combination cleared guardrails.",
                "current_params": {
                    "target_delta": current_params.target_delta,
                    "stop_loss_mult": current_params.stop_loss_mult,
                    "profit_target_pct": current_params.profit_target_pct,
                },
            })
            return

        windows = _make_windows(
            (len(prices) - 1) // DAYS_PER_EXPIRY,
            train_weeks, test_weeks, step_weeks
        )
        baseline_scores = evaluate_params_on_windows(
            prices, current_params, windows, STARTING_CAPITAL, IV_ESTIMATE
        )
        valid_baseline = [s for s in baseline_scores if s != float("-inf")]
        baseline_oos_score = sum(valid_baseline) / len(valid_baseline) if valid_baseline else float("-inf")

        improved = bool(
            candidate.key() != current_params.key()
            and candidate_oos_score > baseline_oos_score + min_imp
        )

        notes = (
            f"{len(window_logs)} walk-forward windows evaluated "
            f"(train={train_weeks}w, test={test_weeks}w, step={step_weeks}w)."
        )

        if auto_adopt:
            cfg.record_optimization_run(
                state, candidate, baseline_oos_score, candidate_oos_score,
                improved=improved, min_improvement=min_imp, notes=notes
            )

        parsed_window_logs = []
        for wl in window_logs:
            parsed_window_logs.append({
                "window": wl.get("window"),
                "train_range": wl.get("train_range"),
                "test_range": wl.get("test_range"),
                "winner_params": {
                    "target_delta": wl["winner_params"].target_delta,
                    "stop_loss_mult": wl["winner_params"].stop_loss_mult,
                    "profit_target_pct": wl["winner_params"].profit_target_pct,
                } if wl.get("winner_params") else None,
                "winner_train_score": round(float(wl.get("winner_train_score", 0)), 3),
                "winner_oos_score": round(float(wl.get("winner_oos_score", 0)), 3),
            })

        self._send_json({
            "success": True,
            "current_params": {
                "target_delta": current_params.target_delta,
                "stop_loss_mult": current_params.stop_loss_mult,
                "profit_target_pct": current_params.profit_target_pct,
            },
            "candidate_params": {
                "target_delta": candidate.target_delta,
                "stop_loss_mult": candidate.stop_loss_mult,
                "profit_target_pct": candidate.profit_target_pct,
            },
            "baseline_oos_score": round(float(baseline_oos_score), 3),
            "candidate_oos_score": round(float(candidate_oos_score), 3),
            "min_improvement_required": min_imp,
            "improved": improved,
            "adopted": improved and auto_adopt,
            "window_logs": parsed_window_logs,
        })


def run_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, BotRequestHandler)
    print(f"Server started on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
