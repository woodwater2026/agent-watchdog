"""
Agent Watchdog — a circuit breaker for AI agent runs.

Usage:
    from agent_watchdog import AgentWatchdog

    watchdog = AgentWatchdog(max_budget_usd=1.0, max_identical_calls=3, timeout_seconds=300)

    with watchdog.watch(run_id="my-run"):
        result = my_agent.run(task)
"""
import time
import hashlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class HaltReason(str, Enum):
    LOOP_DETECTED = "loop_detected"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class HaltReport:
    run_id: str
    reason: HaltReason
    elapsed_seconds: float
    tool_calls: list
    estimated_cost_usd: float
    last_output: Optional[str] = None
    message: str = ""

    def __str__(self):
        return (
            f"[WatchdogHalt] run={self.run_id} reason={self.reason.value} "
            f"elapsed={self.elapsed_seconds:.1f}s cost=${self.estimated_cost_usd:.4f} "
            f"calls={len(self.tool_calls)} msg={self.message}"
        )


class WatchdogHalt(Exception):
    def __init__(self, report: HaltReport):
        self.report = report
        super().__init__(str(report))


@dataclass
class RunState:
    run_id: str
    start_time: float = field(default_factory=time.time)
    tool_calls: list = field(default_factory=list)  # list of (tool_name, args_hash)
    token_in: int = 0
    token_out: int = 0
    last_output: Optional[str] = None
    halted: bool = False
    halt_reason: Optional[HaltReason] = None
    pending_halt: Optional["WatchdogHalt"] = None  # set by background thread


# Approximate pricing (per 1k tokens), same as agent-budget-guard
MODEL_PRICING = {
    "anthropic/claude-sonnet-4-6": {"in": 0.003, "out": 0.015},
    "anthropic/claude-haiku-4-5": {"in": 0.0008, "out": 0.004},
    "openai/gpt-4o": {"in": 0.0025, "out": 0.01},
    "default": {"in": 0.003, "out": 0.015},
}


def _estimate_cost(model: str, token_in: int, token_out: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    return (token_in / 1000) * pricing["in"] + (token_out / 1000) * pricing["out"]


def _args_hash(args: Any) -> str:
    return hashlib.md5(str(args).encode()).hexdigest()[:8]


def _detect_repeating_pattern(window: list) -> list:
    """
    Return the shortest repeating subsequence that fills `window`, or [] if none.

    Example: [A, B, A, B, A, B, A, B] → [A, B]
             [A, B, C, A, B, C] → [A, B, C]
             [A, B, C, D, ...] → [] (no pattern)

    Only patterns of length 2..len(window)//2 are considered (need at least 2 repetitions).
    """
    n = len(window)
    for pat_len in range(2, n // 2 + 1):
        if n % pat_len != 0:
            continue
        pat = window[:pat_len]
        reps = n // pat_len
        if pat * reps == window:
            return pat
    return []


class AgentWatchdog:
    """
    Framework-agnostic circuit breaker for AI agent runs.

    Detects: infinite loops (identical or pattern-based), budget overruns, timeouts.
    On detect: raises WatchdogHalt with a structured report.

    Loop detection modes (both are checked on every call):
      - max_identical_calls: halt if the same (tool, args) appears N times in a row.
        Maps to window_size in the CrewAI #4682 proposal.
      - pattern_window_size: halt if a repeating subsequence fills a window of recent
        calls. Catches ABAB, ABCABC, etc. Set to 0 to disable.
    """

    def __init__(
        self,
        max_budget_usd: float = 1.0,
        max_identical_calls: int = 3,
        timeout_seconds: Optional[float] = 300,
        model: str = "default",
        warn_at_pct: float = 0.8,
        pattern_window_size: int = 8,
    ):
        self.max_budget_usd = max_budget_usd
        self.max_identical_calls = max_identical_calls
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.warn_at_pct = warn_at_pct
        self.pattern_window_size = pattern_window_size
        self._current_run: Optional[RunState] = None
        self._timer: Optional[threading.Timer] = None

    @contextmanager
    def watch(self, run_id: str = "run"):
        """Context manager that monitors the enclosed agent run."""
        state = RunState(run_id=run_id)
        self._current_run = state

        # Start timeout timer
        if self.timeout_seconds:
            self._timer = threading.Timer(
                self.timeout_seconds,
                self._halt,
                args=(state, HaltReason.TIMEOUT, f"exceeded {self.timeout_seconds}s timeout"),
            )
            self._timer.daemon = True
            self._timer.start()

        try:
            yield self
        except WatchdogHalt:
            raise
        except Exception as e:
            state.last_output = f"Exception: {e}"
            raise
        finally:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            # Re-raise any halt that fired from a background thread
            pending = state.pending_halt
            self._current_run = None
            if pending is not None:
                raise pending

    def record_tool_call(self, tool_name: str, args: Any = None, output: Any = None):
        """
        Call this each time the agent invokes a tool.
        Detects identical-consecutive loops and repeating-pattern loops. Thread-safe.
        """
        if self._current_run is None:
            return

        state = self._current_run
        ah = _args_hash(args)
        state.tool_calls.append((tool_name, ah))

        if output is not None:
            state.last_output = str(output)[:500]

        # --- Mode 1: N identical consecutive (tool, args) pairs ---
        if len(state.tool_calls) >= self.max_identical_calls:
            tail = state.tool_calls[-self.max_identical_calls:]
            if len(set(tail)) == 1:
                self._halt(
                    state,
                    HaltReason.LOOP_DETECTED,
                    f"tool '{tool_name}' called {self.max_identical_calls}x identically",
                )

        # --- Mode 2: Sliding-window pattern detection (ABAB, ABCABC, etc.) ---
        # Looks for the shortest repeating subsequence that perfectly fills a window.
        if self.pattern_window_size >= 4 and len(state.tool_calls) >= self.pattern_window_size:
            window = state.tool_calls[-self.pattern_window_size:]
            pattern = _detect_repeating_pattern(window)
            if pattern:
                self._halt(
                    state,
                    HaltReason.LOOP_DETECTED,
                    f"repeating pattern detected over last {self.pattern_window_size} calls "
                    f"(pattern length {len(pattern)}): {[t for t, _ in pattern]}",
                )

    def record_tokens(self, token_in: int = 0, token_out: int = 0):
        """
        Call this to update token usage mid-run.
        Checks budget and halts if exceeded.
        """
        if self._current_run is None:
            return

        state = self._current_run
        state.token_in += token_in
        state.token_out += token_out

        cost = _estimate_cost(self.model, state.token_in, state.token_out)

        if cost >= self.max_budget_usd:
            self._halt(
                state,
                HaltReason.BUDGET_EXCEEDED,
                f"cost ${cost:.4f} exceeded limit ${self.max_budget_usd}",
            )

        warn_threshold = self.max_budget_usd * self.warn_at_pct
        if cost >= warn_threshold:
            print(
                f"[Watchdog WARNING] run={state.run_id} cost=${cost:.4f} "
                f"({self.warn_at_pct*100:.0f}% of ${self.max_budget_usd} budget)"
            )

    def halt(self, reason: str = "manual stop"):
        """Manually halt the current run."""
        if self._current_run:
            self._halt(self._current_run, HaltReason.MANUAL, reason)

    def _halt(self, state: RunState, reason: HaltReason, message: str):
        if state.halted:
            return
        state.halted = True
        state.halt_reason = reason

        elapsed = time.time() - state.start_time
        cost = _estimate_cost(self.model, state.token_in, state.token_out)

        report = HaltReport(
            run_id=state.run_id,
            reason=reason,
            elapsed_seconds=elapsed,
            tool_calls=list(state.tool_calls),
            estimated_cost_usd=cost,
            last_output=state.last_output,
            message=message,
        )
        exc = WatchdogHalt(report)
        print(f"[Watchdog HALT] {report}")
        # If we're in the main thread, raise directly; otherwise store for context manager
        if threading.current_thread() is threading.main_thread():
            raise exc
        else:
            state.pending_halt = exc
