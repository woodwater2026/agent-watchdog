"""
Agent Watchdog + LangChain Integration
=======================================
Protect your LangChain agents from infinite loops, budget overruns, and timeouts.

Quick start (no API key needed):
    pip install agent-watchdog
    python langchain_example.py           # runs four self-contained demos

Full integration (requires langchain):
    pip install agent-watchdog langchain langchain-openai
    # then use WatchdogCallback + run_agent_safely() — see sections 2–3 below.

What this file covers:
  1. Watchdog configuration reference
  2. WatchdogCallback  — LangChain callback handler (feeds tool calls + tokens)
  3. run_agent_safely() — drop-in wrapper for any AgentExecutor or LCEL Runnable
  4. print_halt_summary() — human-readable halt report with actionable advice
  5. Four standalone demos: loop detection, budget guard, ABAB pattern, timeout
  6. Patterns: reusing the watchdog, inspecting tool-call traces
  7. REAL_AGENT_EXAMPLE — a complete copy-paste setup for production use

Beginner? Start with examples/langchain_quickstart.py instead.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agent_watchdog import AgentWatchdog, HaltReason, WatchdogHalt


# ---------------------------------------------------------------------------
# 1. Watchdog configuration
#
#    One watchdog instance per agent type. It is safe to reuse across multiple
#    .watch() calls — each call gets an independent run state.
# ---------------------------------------------------------------------------
watchdog = AgentWatchdog(
    max_budget_usd=0.50,        # halt if estimated cost exceeds $0.50
    max_identical_calls=3,      # halt if same tool+args called 3x in a row
    pattern_window_size=8,      # also catch ABAB / ABCABC repeating patterns (0 = off)
    timeout_seconds=120,        # absolute wall-clock limit (2 min)
    model="openai/gpt-4o",      # used for cost estimation — no network call made
    warn_at_pct=0.8,            # print a warning at 80% budget consumption
)


# ---------------------------------------------------------------------------
# 2. WatchdogCallback — LangChain callback handler
#
#    Wires into LangChain's callback system to:
#      - Record every tool invocation for loop / pattern detection
#      - Extract token usage from LLM responses for budget tracking
#
#    Thread safety: LangChain may invoke callbacks from multiple threads when
#    using async chains. AgentWatchdog's internal state is protected by a lock,
#    so WatchdogCallback is safe for concurrent use.
#
#    Usage:
#        callback = WatchdogCallback(watchdog)
#        # AgentExecutor:
#        result = agent_executor.invoke({"input": task}, config={"callbacks": [callback]})
#        # LCEL chain:
#        result = (prompt | llm | parser).invoke(inputs, config={"callbacks": [callback]})
# ---------------------------------------------------------------------------
try:
    from langchain.callbacks.base import BaseCallbackHandler  # type: ignore

    class WatchdogCallback(BaseCallbackHandler):
        """
        LangChain callback that feeds tool calls and token usage into AgentWatchdog.

        Plug this into any AgentExecutor, LLMChain, or LCEL Runnable:

            callback = WatchdogCallback(watchdog)
            config = {"callbacks": [callback]}
            result = agent_executor.invoke({"input": task}, config=config)
        """

        def __init__(self, wd: AgentWatchdog) -> None:
            super().__init__()
            self.watchdog = wd

        # ── Tool hooks ──────────────────────────────────────────────────────

        def on_tool_start(
            self,
            serialized: dict,
            input_str: str,
            **kwargs: Any,
        ) -> None:
            """
            Called before a tool runs. Records the call for loop/pattern detection.

            Raises WatchdogHalt immediately if a loop is detected; LangChain
            propagates this exception up through the AgentExecutor.
            """
            tool_name = serialized.get("name", "unknown_tool")
            self.watchdog.record_tool_call(tool_name, args=input_str)

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            """
            Called after a tool returns. Saves the output for the halt report.

            Note: `_current_run` is an internal attribute. It is `None` outside
            of an active `watchdog.watch()` context.
            """
            if self.watchdog._current_run is not None:
                # Truncate to 500 chars — avoids bloating the halt report
                self.watchdog._current_run.last_output = str(output)[:500]

        def on_tool_error(
            self,
            error: BaseException,
            **kwargs: Any,
        ) -> None:
            """
            Called when a tool raises an exception. Records the error as the
            last output so halt reports include it for easier debugging.
            """
            if self.watchdog._current_run is not None:
                self.watchdog._current_run.last_output = (
                    f"ToolError: {type(error).__name__}: {error}"
                )

        # ── LLM hooks ───────────────────────────────────────────────────────

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            """
            Called after each LLM response. Extracts token usage for budget tracking.

            Works with OpenAI, Anthropic, and any provider that populates
            response.llm_output["token_usage"] (OpenAI) or
            response.llm_output["usage"] (Anthropic).
            """
            llm_output = getattr(response, "llm_output", None) or {}

            # Try OpenAI key first, then fall back to Anthropic's key name
            usage: dict = llm_output.get("token_usage") or llm_output.get("usage") or {}

            token_in: int = (
                usage.get("prompt_tokens")      # OpenAI
                or usage.get("input_tokens")    # Anthropic
                or 0
            )
            token_out: int = (
                usage.get("completion_tokens")  # OpenAI
                or usage.get("output_tokens")   # Anthropic
                or 0
            )

            if token_in or token_out:
                # Raises WatchdogHalt if this token batch pushes cost over the limit
                self.watchdog.record_tokens(token_in=token_in, token_out=token_out)

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            """Called on LLM API errors. Logs them; no watchdog action needed here."""
            print(f"[Watchdog] LLM error observed: {type(error).__name__}: {error}")

except ImportError:
    WatchdogCallback = None  # type: ignore[misc,assignment]
    print("[Watchdog] LangChain not installed — WatchdogCallback unavailable.")
    print("           Install with: pip install langchain")


# ---------------------------------------------------------------------------
# 3. run_agent_safely() — drop-in replacement for agent_executor.invoke()
#
#    Returns (result, None)   on success
#    Returns (None, report)   if the watchdog halted the run
#
#    Accepts an optional `wd` parameter so you can pass a custom watchdog
#    instance instead of using the module-level one — useful when your app
#    manages multiple agents with different budget limits.
# ---------------------------------------------------------------------------
def run_agent_safely(
    agent_executor: Any,
    task: str,
    run_id: str = "langchain-run",
    extra_callbacks: Optional[list] = None,
    wd: Optional[AgentWatchdog] = None,
) -> tuple[Optional[dict], Optional[Any]]:
    """
    Run a LangChain AgentExecutor with full watchdog protection.

    Args:
        agent_executor:  A LangChain AgentExecutor (or any LCEL Runnable).
        task:            The user's task string passed as {"input": task}.
        run_id:          Unique identifier for this run (used in logs and reports).
        extra_callbacks: Additional LangChain callbacks to include alongside
                         WatchdogCallback (e.g. LangSmith tracing callbacks).
        wd:              Optional AgentWatchdog instance. Defaults to the
                         module-level `watchdog` if not provided.

    Returns:
        (result, None)  — successful completion; `result` is the agent's output dict.
        (None, report)  — watchdog halted the run; `report` is a HaltReport with
                          fields: reason, elapsed_seconds, estimated_cost_usd,
                          tool_calls, last_output, message.

    Raises:
        ImportError:  if langchain is not installed.
        Exception:    re-raises any unexpected non-watchdog exception from the agent.

    Example:
        result, report = run_agent_safely(executor, "Summarize this topic")
        if report:
            print(f"Agent halted: {report.reason} — cost ${report.estimated_cost_usd:.4f}")
        else:
            print(result["output"])
    """
    if WatchdogCallback is None:
        raise ImportError("LangChain is required. Install with: pip install langchain")

    active_wd = wd or watchdog  # use provided instance or fall back to module-level

    callbacks = [WatchdogCallback(active_wd)] + (extra_callbacks or [])

    try:
        with active_wd.watch(run_id=run_id):
            result = agent_executor.invoke(
                {"input": task},
                config={"callbacks": callbacks},
            )
            return result, None

    except WatchdogHalt as e:
        # Watchdog fired — print a human-readable summary and return the report
        print_halt_summary(e.report)
        return None, e.report

    except KeyboardInterrupt:
        # Ctrl-C during a run: let it propagate naturally
        print(f"\n[Watchdog] Run '{run_id}' interrupted by user.")
        raise

    except Exception as e:
        # Unexpected agent / tool error — log context and re-raise
        print(f"[Watchdog] Unexpected error during run '{run_id}': {type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# 4. print_halt_summary() — human-readable halt report with fix suggestions
# ---------------------------------------------------------------------------

def print_halt_summary(r: Any) -> None:
    """
    Pretty-print a HaltReport with the halt reason, cost, call count, and
    a concrete suggested fix.

    Args:
        r: A HaltReport returned in e.report from a WatchdogHalt exception.
    """
    print(f"\n{'='*60}")
    print(f"[Watchdog] Run halted: {r.run_id}")
    print(f"  Reason:   {r.reason.value}")
    print(f"  Elapsed:  {r.elapsed_seconds:.1f}s")
    print(f"  Cost:     ${r.estimated_cost_usd:.4f}")
    print(f"  Calls:    {len(r.tool_calls)}")
    if r.last_output:
        print(f"  Last out: {r.last_output[:120]}")

    if r.reason == HaltReason.LOOP_DETECTED:
        print(f"  → Agent is stuck repeating tool calls with no progress.")
        print(f"    Fix: retry with a reflection prompt, switch tools, or increase max_identical_calls.")
    elif r.reason == HaltReason.BUDGET_EXCEEDED:
        print(f"  → Token cost hit the limit. Increase max_budget_usd or simplify the task.")
    elif r.reason == HaltReason.TIMEOUT:
        print(f"  → Wall-clock limit reached. Increase timeout_seconds or break into sub-tasks.")
    elif r.reason == HaltReason.MANUAL:
        print(f"  → Run stopped manually via watchdog.request_halt().")

    print(f"{'='*60}\n")


# Backward-compatible alias (was private in earlier versions)
_print_halt_summary = print_halt_summary


# ---------------------------------------------------------------------------
# 5. Standalone demos — run without LangChain or an API key
#    These simulate a LangChain agent's step-by-step behavior directly,
#    so you can observe the watchdog's behavior without writing any agent code.
# ---------------------------------------------------------------------------

def _demo_loop_detection() -> None:
    """
    Simulate a LangChain agent stuck calling the same search tool repeatedly.

    This mirrors what happens when a web-search tool returns unhelpful results
    and the LLM keeps retrying the exact same query — a common failure mode.

    In a real agent, WatchdogCallback.on_tool_start() fires record_tool_call()
    automatically. Here we call it directly to show the raw mechanics.
    """
    print("\n" + "="*60)
    print("DEMO 1: Loop Detection — Identical Consecutive Calls")
    print("="*60)
    print("Simulating agent stuck in: search('how to fix X') × ∞")
    print()

    demo_wd = AgentWatchdog(
        max_budget_usd=0.50,
        max_identical_calls=3,   # halt after 3 back-to-back identical calls
        timeout_seconds=30,
        model="openai/gpt-4o",
    )

    try:
        with demo_wd.watch(run_id="demo-loop"):
            for i in range(10):
                tool_name = "tavily_search_results_json"
                args = {"query": "how to fix X error in Python"}
                output = "Error: no results found. Try a different query."

                print(f"  Step {i+1}: agent calls {tool_name!r} → '{output[:40]}...'")

                # Simulate the token cost the LLM would charge per step
                demo_wd.record_tokens(token_in=800, token_out=150)

                # This is what WatchdogCallback.on_tool_start() does internally.
                # Raises WatchdogHalt on the 3rd identical call.
                demo_wd.record_tool_call(tool_name, args=args, output=output)

                time.sleep(0.02)

    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ Loop caught after {len(r.tool_calls)} tool calls (would have run 10+)")
        print(f"✓ Reason:  {r.reason.value}")
        print(f"✓ Elapsed: {r.elapsed_seconds:.2f}s")
        print(f"✓ Cost at halt: ${r.estimated_cost_usd:.4f}")
        print(f"✓ Inspect the full call trace: e.report.tool_calls (list of ToolCallRecord)")


def _demo_budget_guard() -> None:
    """
    Simulate a multi-step research agent that exhausts its token budget.

    A typical research agent might run 15–20 steps with long prompts.
    The watchdog stops it before the bill escalates.

    The warn_at_pct=0.8 setting emits a warning at 80% budget consumption
    so you can see it before the hard halt fires.
    """
    print("\n" + "="*60)
    print("DEMO 2: Budget Guard — Token Cost Overrun")
    print("="*60)

    BUDGET = 0.02   # tight budget so it trips quickly in the demo
    demo_wd = AgentWatchdog(
        max_budget_usd=BUDGET,
        max_identical_calls=20,  # not testing loop detection here
        timeout_seconds=60,
        model="openai/gpt-4o",
        warn_at_pct=0.8,         # warn when 80% of $0.02 is spent (~$0.016)
    )

    tools_sequence = [
        ("tavily_search_results_json", "climate change solutions"),
        ("wikipedia",                  "carbon capture technology"),
        ("tavily_search_results_json", "renewable energy statistics 2024"),
        ("read_webpage",               "https://example.com/energy-report"),
        ("python_repl",                "import pandas; df.describe()"),
        ("tavily_search_results_json", "solar panel efficiency records"),
    ]

    print(f"Budget: ${BUDGET}. Running multi-tool research agent...")
    print()

    try:
        with demo_wd.watch(run_id="demo-budget"):
            # Repeat the sequence to ensure we hit the budget ceiling
            for i, (tool, args) in enumerate(tools_sequence * 5):
                args_preview = repr(args)[:40]
                print(f"  Step {i+1}: {tool}({args_preview})")

                # Simulate realistic token usage: 1500 in, 400 out per step
                # record_tokens() raises WatchdogHalt when cumulative cost > BUDGET
                demo_wd.record_tokens(token_in=1500, token_out=400)
                demo_wd.record_tool_call(tool, args=args)
                time.sleep(0.01)

    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ Budget guard fired after {len(r.tool_calls)} tool calls")
        print(f"✓ Cost at halt:  ${r.estimated_cost_usd:.4f} (limit was ${BUDGET})")
        print(f"✓ Reason: {r.reason.value}")
        print(f"✓ The agent would have kept going — watchdog saved the overage")


def _demo_pattern_detection() -> None:
    """
    Simulate an agent alternating between two tools with no progress (ABAB loop).

    max_identical_calls only catches exact repeats (AAAA). The sliding-window
    pattern detector catches harder-to-spot patterns like ABABABAB or ABCABC.

    Here the agent calls 'search' then 'read_page' in a round-robin forever.
    The watchdog fires when the last `pattern_window_size` calls form a
    repeating subsequence — no identical consecutive calls required.
    """
    print("\n" + "="*60)
    print("DEMO 3: Sliding-Window Pattern Detection (ABAB / ABCABC loops)")
    print("="*60)
    print("Simulating agent alternating: search → read_page → search → read_page → ...")
    print()

    demo_wd = AgentWatchdog(
        max_budget_usd=0.50,
        max_identical_calls=10,  # would NOT fire on alternating calls
        pattern_window_size=8,   # fires when last 8 calls form a repeating pattern
        timeout_seconds=30,
        model="openai/gpt-4o",
    )

    try:
        with demo_wd.watch(run_id="demo-abab"):
            for i in range(12):
                # Alternating tools, same args each round — classic ABAB pattern
                if i % 2 == 0:
                    tool, args = "tavily_search_results_json", {"query": "Python async tutorial"}
                else:
                    tool, args = "read_webpage", {"url": "https://docs.python.org/async"}

                print(f"  Step {i+1}: {tool}(...)")
                demo_wd.record_tokens(token_in=600, token_out=200)
                demo_wd.record_tool_call(tool, args=args)
                time.sleep(0.02)

    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ ABAB pattern caught after {len(r.tool_calls)} calls")
        print(f"✓ Reason:  {r.reason.value}")
        print(f"✓ Message: {r.message}")
        print(f"  (max_identical_calls alone would NOT have caught this)")


def _demo_timeout() -> None:
    """
    Simulate a slow agent that exceeds the wall-clock timeout.

    The watchdog runs a background timer thread. When it fires, it sets a
    `pending_halt` flag on the run state. The next call to record_tool_call()
    or record_tokens() checks the flag and raises WatchdogHalt.

    Best practice: also check `pending_halt` between steps (shown below) so
    a slow step can exit cleanly without waiting for the next record_* call.
    """
    print("\n" + "="*60)
    print("DEMO 4: Timeout — Wall-Clock Limit")
    print("="*60)

    demo_wd = AgentWatchdog(
        max_budget_usd=10.0,
        max_identical_calls=50,
        timeout_seconds=0.5,    # very short for demo purposes
        model="openai/gpt-4o",
    )

    print("Running a 'slow' agent with a 0.5s timeout...")
    print()

    try:
        with demo_wd.watch(run_id="demo-timeout"):
            for i in range(20):
                # Proactive check: the background timer sets pending_halt but
                # can't interrupt a blocking step. Checking here lets us exit
                # cleanly between steps rather than waiting for the next record_*
                # call to raise the exception.
                state = demo_wd._current_run
                if state and state.pending_halt:
                    break

                print(f"  Step {i+1}: agent thinking... (slow LLM call)")
                demo_wd.record_tokens(token_in=300, token_out=100)
                demo_wd.record_tool_call(f"tool_{i}", args=f"step_{i}")
                time.sleep(0.2)   # each step takes 200ms → times out after ~2–3 steps

    except WatchdogHalt as e:
        r = e.report
        print(f"\n✓ Timeout halt after {r.elapsed_seconds:.2f}s")
        print(f"✓ Completed {len(r.tool_calls)} tool calls before halt")
        print(f"✓ Reason: {r.reason.value}")


# ---------------------------------------------------------------------------
# 6. Patterns: reusing the watchdog and inspecting tool-call traces
#
#    These are not runnable demos — they show common production patterns.
# ---------------------------------------------------------------------------

def _pattern_reuse_watchdog() -> None:
    """
    Pattern: run multiple tasks through the same watchdog instance.

    Each call to watchdog.watch() creates a fresh run state, so prior run
    history does not carry over. The watchdog configuration (budget, limits,
    model) is shared and not reset between runs.
    """
    tasks = ["Research topic A", "Summarize topic B", "Analyze topic C"]

    for i, task in enumerate(tasks):
        try:
            with watchdog.watch(run_id=f"batch-run-{i}"):
                # ... run your agent here ...
                pass
        except WatchdogHalt as e:
            print(f"Task {i} halted: {e.report.reason.value}")
            # Continue to next task — watchdog is ready for a new run
            continue


def _pattern_inspect_trace(report: Any) -> None:
    """
    Pattern: inspect the tool-call trace from a HaltReport.

    Each entry in report.tool_calls is a ToolCallRecord with:
      - tool_name: str
      - args: Any
      - output: str | None
      - timestamp: float (unix epoch)
    """
    print(f"Run '{report.run_id}' made {len(report.tool_calls)} tool calls:")
    for i, call in enumerate(report.tool_calls):
        print(f"  [{i+1}] {call.tool_name}({repr(call.args)[:60]})")
        if call.output:
            print(f"       → {call.output[:80]}")


# ---------------------------------------------------------------------------
# 7. Real LangChain agent example (requires langchain + langchain-openai + OPENAI_API_KEY)
#
#    Copy-paste this into your project for a production-ready setup.
#    The snippet below is intentionally kept as a string so this file
#    can be imported without langchain being installed.
# ---------------------------------------------------------------------------

REAL_AGENT_EXAMPLE = '''
# ── Real LangChain Agent (requires: pip install langchain langchain-openai) ──
#
# This is a complete, production-ready setup. Copy-paste and replace the tools
# and task with your own.

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain import hub
from agent_watchdog import AgentWatchdog, WatchdogHalt
from langchain_example import WatchdogCallback, run_agent_safely, print_halt_summary

# 1. Define your tools
def search_web(query: str) -> str:
    """Search the web for information."""
    # Replace with: from langchain_community.tools import TavilySearchResults
    return f"Search result for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    try:
        # NOTE: eval() is used here only for demo purposes.
        # In production, use a proper math library (e.g. sympy, numexpr).
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(name="search",     func=search_web, description="Search the web for information"),
    Tool(name="calculator", func=calculate,  description="Evaluate a math expression"),
]

# 2. Create the LLM + agent
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ["OPENAI_API_KEY"])
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. Configure the watchdog
#    Use a separate instance per agent type so each has its own budget.
my_watchdog = AgentWatchdog(
    max_budget_usd=0.50,
    max_identical_calls=3,
    pattern_window_size=8,
    timeout_seconds=120,
    model="openai/gpt-4o",
    warn_at_pct=0.8,
)

# 4. Run safely
result, report = run_agent_safely(
    agent_executor,
    task="What is the capital of France and what is 42 * 1337?",
    run_id="my-research-run",
    wd=my_watchdog,    # pass your own watchdog instead of the module-level default
)

if report:
    # Watchdog halted the run — inspect the structured report
    print_halt_summary(report)

    # Access the full tool-call trace for logging or observability
    for call in report.tool_calls:
        print(f"  {call.tool_name}({repr(call.args)[:60]})")

    # Optionally retry with a rephrased prompt or escalate to a human
else:
    print("Success:", result["output"])
'''


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Agent Watchdog + LangChain — standalone demos")
    print("(No API key required for these demos)\n")

    _demo_loop_detection()
    _demo_budget_guard()
    _demo_pattern_detection()
    _demo_timeout()

    print("\n" + "="*60)
    print("All demos complete.\n")
    print("To use with a real LangChain agent:")
    print()
    print("  pip install langchain langchain-openai")
    print()
    print("  from langchain_example import WatchdogCallback, run_agent_safely")
    print()
    print("  result, report = run_agent_safely(your_executor, 'Your task here')")
    print("  if report:")
    print("      print(f'Halted: {report.reason} — cost ${report.estimated_cost_usd:.4f}')")
    print("  else:")
    print("      print(result['output'])")
    print()
    print("See REAL_AGENT_EXAMPLE in this file for a complete production setup.")
    print("See examples/langchain_quickstart.py for a minimal beginner intro.")
    print("="*60)
