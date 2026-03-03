"""
Agent Watchdog + CrewAI Integration
====================================
Addresses the problem described in CrewAI issues #4495 and #4682:
  - Agents entering infinite tool-call loops
  - No built-in behavioral loop detection / middleware

Agent Watchdog solves this NOW, without waiting for a framework update.

Quick start:
    pip install agent-watchdog crewai
    python crewai_example.py          # runs a self-contained loop demo (no API key needed)

Full crew usage: see run_crew_safely() below.
"""

from __future__ import annotations

import time
from typing import Any

from agent_watchdog import AgentWatchdog, HaltReason, WatchdogHalt

# ---------------------------------------------------------------------------
# 1. Watchdog configuration
#    Maps directly to the API proposed in issue #4682:
#
#    window_size       → max_identical_calls (sliding window of repeated calls)
#    similarity_threshold → (exact-match by default; see note on fuzzy matching below)
#    on_loop           → raises WatchdogHalt — handle it to inject a reflection prompt,
#                        stop the crew, escalate, etc.
# ---------------------------------------------------------------------------
watchdog = AgentWatchdog(
    max_budget_usd=1.0,         # halt if estimated cost exceeds $1
    max_identical_calls=3,      # halt if same tool+args seen 3x in a row  ← window_size
    timeout_seconds=300,        # absolute wall-clock limit (5 min)
    model="anthropic/claude-sonnet-4-6",
    warn_at_pct=0.8,            # log a warning at 80% budget consumption
)


# ---------------------------------------------------------------------------
# 2. CrewAI tool-level integration via BaseTool monkey-patch
#    (Uncomment in your project — kept commented here for portability)
#
#    Why: CrewAI doesn't expose a universal middleware hook yet (#4682).
#    This approach patches at the BaseTool level so every tool is covered
#    automatically, including third-party tools.
# ---------------------------------------------------------------------------
def install_crewai_patch(wd: AgentWatchdog) -> None:
    """
    Monkey-patch crewai.tools.BaseTool._run so every tool call is recorded
    by the watchdog. Call this once before crew.kickoff().

    This is equivalent to the LoopDetector middleware proposed in #4682.
    """
    try:
        from crewai.tools import BaseTool  # type: ignore

        _orig = BaseTool._run

        def _patched_run(self, *args: Any, **kwargs: Any) -> Any:
            result = _orig(self, *args, **kwargs)
            # Record the call AFTER execution so we have the output hash too
            wd.record_tool_call(
                tool_name=self.name,
                args={"args": args, "kwargs": kwargs},
                output=result,
            )
            return result

        BaseTool._run = _patched_run  # type: ignore[method-assign]
        print("[Watchdog] CrewAI BaseTool patched — all tool calls now monitored.")
    except ImportError:
        print("[Watchdog] crewai not installed; skipping BaseTool patch.")


# ---------------------------------------------------------------------------
# 3. Per-step callback (alternative / complementary to the monkey-patch)
#    Pass this as step_callback= to Crew() for per-iteration awareness.
# ---------------------------------------------------------------------------
def make_step_callback(wd: AgentWatchdog):
    """
    Returns a CrewAI step_callback that feeds token usage into the watchdog.

    Usage:
        crew = Crew(
            agents=[...],
            tasks=[...],
            step_callback=make_step_callback(watchdog),
        )
    """
    def _step_callback(step_output: Any) -> None:
        # CrewAI passes an AgentFinish or similar object here.
        # Extract token usage when available.
        usage = getattr(step_output, "token_usage", None)
        if usage:
            wd.record_tokens(
                token_in=getattr(usage, "prompt_tokens", 0),
                token_out=getattr(usage, "completion_tokens", 0),
            )

    return _step_callback


# ---------------------------------------------------------------------------
# 4. Main run wrapper — drop-in replacement for crew.kickoff()
# ---------------------------------------------------------------------------
def run_crew_safely(
    crew,                        # your crewai.Crew instance
    inputs: dict,
    run_id: str = "crewai-run",
    on_loop: str = "stop",       # "stop" | "inject_reflection" | "escalate"
) -> dict:
    """
    Run a CrewAI crew with full watchdog protection.

    Handles the three failure modes from issue #4682:
      - loop_detected   → repeated tool calls without progress
      - budget_exceeded → token cost overrun
      - timeout         → wall-clock limit reached

    on_loop controls behavior when a loop is detected:
      "stop"              → return an error dict (safe default)
      "inject_reflection" → restart the crew with an added reflection prompt
      "escalate"          → re-raise so your orchestrator can handle it
    """
    install_crewai_patch(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff(inputs=inputs)
            return {"status": "ok", "result": result}

    except WatchdogHalt as e:
        r = e.report
        _print_halt_summary(r)

        if r.reason == HaltReason.LOOP_DETECTED:
            if on_loop == "inject_reflection":
                return _retry_with_reflection(crew, inputs, run_id, r)
            elif on_loop == "escalate":
                raise

        # Default: return structured error dict
        return {
            "status": "halted",
            "reason": r.reason.value,
            "elapsed_seconds": r.elapsed_seconds,
            "estimated_cost_usd": r.estimated_cost_usd,
            "tool_calls": len(r.tool_calls),
            "last_output": r.last_output,
        }


def _retry_with_reflection(crew, inputs: dict, run_id: str, report) -> dict:
    """
    Re-run the crew with a reflection prompt injected.
    Mirrors the `on_loop="inject_reflection"` behavior from issue #4682.
    """
    reflection = (
        "SYSTEM: You appear to be repeating the same actions without progress. "
        "Step back, reconsider your approach, and try a different strategy. "
        f"Repeated action: {report.tool_calls[-1] if report.tool_calls else 'unknown'}"
    )
    print(f"\n[Watchdog] Injecting reflection prompt and retrying...")
    enriched_inputs = {**inputs, "system_context": reflection}

    # Fresh watchdog for the retry
    retry_wd = AgentWatchdog(
        max_budget_usd=watchdog.max_budget_usd,
        max_identical_calls=watchdog.max_identical_calls,
        timeout_seconds=watchdog.timeout_seconds,
        model=watchdog.model,
    )
    try:
        with retry_wd.watch(run_id=f"{run_id}-retry"):
            result = crew.kickoff(inputs=enriched_inputs)
            return {"status": "ok_after_reflection", "result": result}
    except WatchdogHalt as e:
        _print_halt_summary(e.report)
        return {"status": "halted_after_reflection", "reason": e.report.reason.value}


def _print_halt_summary(r) -> None:
    print(f"\n{'='*60}")
    print(f"[Watchdog] Run halted: {r.run_id}")
    print(f"  Reason:   {r.reason.value}")
    print(f"  Elapsed:  {r.elapsed_seconds:.1f}s")
    print(f"  Cost:     ${r.estimated_cost_usd:.4f}")
    print(f"  Calls:    {len(r.tool_calls)}")
    if r.last_output:
        print(f"  Last out: {r.last_output[:120]}")
    if r.reason == HaltReason.LOOP_DETECTED:
        print(f"  → Agent repeated tool calls. Loop broken.")
        print(f"    Consider: different tool, more specific input, or max_iter limit.")
    elif r.reason == HaltReason.BUDGET_EXCEEDED:
        print(f"  → Cost limit hit. Increase max_budget_usd or simplify the task.")
    elif r.reason == HaltReason.TIMEOUT:
        print(f"  → Time limit hit. Increase timeout_seconds or break task into steps.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 5. Standalone demo — runs without crewai or API keys
#    Simulates the exact failure mode described in issue #4682
# ---------------------------------------------------------------------------
def _demo_loop_detection() -> None:
    """
    Simulate an agent stuck in a loop calling the same search tool repeatedly.
    Demonstrates that AgentWatchdog catches it before it costs a fortune.
    """
    print("\n" + "="*60)
    print("DEMO: Agent Watchdog — Loop Detection")
    print("Simulating the failure mode from CrewAI issue #4682")
    print("="*60)

    demo_wd = AgentWatchdog(
        max_budget_usd=0.10,
        max_identical_calls=3,  # trip after 3 identical calls
        timeout_seconds=30,
        model="anthropic/claude-sonnet-4-6",
    )

    class FakeLoopingAgent:
        """Simulates an agent stuck searching for the same thing."""
        def run(self, task: str, wd: AgentWatchdog) -> str:
            print(f"\nAgent starting task: {task!r}")
            for i in range(10):
                print(f"  Step {i+1}: calling web_search('how to fix X')")
                # Simulate token usage per step
                wd.record_tokens(token_in=500, token_out=200)
                # Record the tool call — same args every time → triggers loop detection
                wd.record_tool_call(
                    tool_name="web_search",
                    args={"query": "how to fix X"},
                    output=f"Result {i}: no useful info found",
                )
                time.sleep(0.05)
            return "done"  # never reached

    agent = FakeLoopingAgent()

    try:
        with demo_wd.watch(run_id="demo-loop"):
            agent.run("Research topic X", wd=demo_wd)
    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ Loop caught after {len(r.tool_calls)} tool calls")
        print(f"✓ Elapsed: {r.elapsed_seconds:.2f}s")
        print(f"✓ Estimated cost saved: would have run 10x, stopped at {len(r.tool_calls)}x")
        print(f"✓ Reason: {r.reason.value}")


def _demo_budget_guard() -> None:
    """Simulate a run that exceeds its token budget."""
    print("\n" + "="*60)
    print("DEMO: Agent Watchdog — Budget Guard")
    print("="*60)

    demo_wd = AgentWatchdog(
        max_budget_usd=0.01,   # tight budget to trigger quickly
        max_identical_calls=10,
        timeout_seconds=30,
        model="anthropic/claude-sonnet-4-6",
    )

    try:
        with demo_wd.watch(run_id="demo-budget"):
            for i in range(20):
                demo_wd.record_tokens(token_in=500, token_out=300)
                demo_wd.record_tool_call(f"tool_{i}", args=i)
    except WatchdogHalt as e:
        r = e.report
        print(f"✓ Budget guard fired after {len(r.tool_calls)} calls")
        print(f"✓ Cost at halt: ${r.estimated_cost_usd:.4f} (limit was $0.01)")


def _demo_pattern_loop() -> None:
    """
    Simulate an ABAB alternating pattern — a subtler loop that's also problematic.
    Agent Watchdog detects repeating subsequences (ABAB, ABCABC, etc.) via
    sliding-window pattern detection in addition to identical-consecutive detection.
    """
    print("\n" + "="*60)
    print("DEMO: Agent Watchdog — Sliding-Window Pattern Detection (ABAB / ABCABC)")
    print("="*60)

    demo_wd = AgentWatchdog(
        max_budget_usd=0.50,
        max_identical_calls=10,  # wouldn't fire on ABAB alone
        pattern_window_size=8,   # fires when last 8 calls form a repeating pattern
        timeout_seconds=10,
        model="default",
    )

    print("Simulating agent alternating between 'search' and 'read_page' with no progress...")
    try:
        with demo_wd.watch(run_id="demo-abab"):
            for i in range(10):
                tool = "search" if i % 2 == 0 else "read_page"
                print(f"  Step {i+1}: {tool}('same_query')")
                demo_wd.record_tokens(token_in=300, token_out=150)
                demo_wd.record_tool_call(tool, args="same_query")
    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ ABAB pattern caught after {len(r.tool_calls)} calls")
        print(f"✓ Reason: {r.reason.value}")
        print(f"✓ Message: {r.message}")


if __name__ == "__main__":
    _demo_loop_detection()
    _demo_budget_guard()
    _demo_pattern_loop()

    print("\n" + "="*60)
    print("To use with a real CrewAI crew:")
    print()
    print("  from crewai import Agent, Task, Crew")
    print("  from examples.crewai_example import run_crew_safely")
    print()
    print("  crew = Crew(agents=[...], tasks=[...])")
    print("  result = run_crew_safely(crew, inputs={'topic': 'X'})")
    print()
    print("  # Or with reflection on loop:")
    print("  result = run_crew_safely(crew, inputs=..., on_loop='inject_reflection')")
    print("="*60)
