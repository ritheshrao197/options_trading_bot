"""
Persistent state for the self-improvement loop.

strategy_config.json holds:
  - "params": the strategy parameters currently considered "live"
  - "history": an append-only audit log of every optimization run --
    whether it changed anything, what it tried, and why

This is what makes the project's improvement *persistent* rather than a
one-off script: main.py's normal run mode always reads the current
params from here, so once optimizer.py updates this file, every
subsequent run (backtest, paper, or live) automatically uses the
improved parameters -- no manual code edits.
"""

import json
import os
import time
from dataclasses import asdict

from backtest import StrategyParams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "strategy_config.json")

DEFAULT_STATE = {
    "params": asdict(StrategyParams()),
    "history": [],
}


def load_state() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_state(DEFAULT_STATE)
        return json.loads(json.dumps(DEFAULT_STATE))
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_live_params(state: dict) -> StrategyParams:
    return StrategyParams(**state["params"])


def record_optimization_run(state: dict, candidate: StrategyParams,
                             baseline_oos_score: float, candidate_oos_score: float,
                             improved: bool, min_improvement: float, notes: str = "") -> dict:
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_params": state["params"],
        "candidate_params": asdict(candidate),
        "baseline_oos_score": round(baseline_oos_score, 4) if baseline_oos_score == baseline_oos_score else None,
        "candidate_oos_score": round(candidate_oos_score, 4) if candidate_oos_score == candidate_oos_score else None,
        "min_improvement_required": min_improvement,
        "adopted": improved,
        "notes": notes,
    }
    state["history"].append(entry)
    if improved:
        state["params"] = asdict(candidate)
    save_state(state)
    return state
