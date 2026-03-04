"""
Agent Watchdog + LangChain — 5-Minute Quick Start
==================================================
The minimal integration. Copy-paste this to protect any LangChain agent.

Installation:
    pip install agent-watchdog

No API key required — this file runs a mock agent to demonstrate the concept.
For a complete, production-ready example see langchain_example.py.
"""

from __future__ import annotations

from agent_watchdog import AgentWatchdog, HaltReason, WatchdogHalt

# ── 1. Configure the watchdog ─────────────────────────────────────────────────
# One watchdog instance per agent type. Reuse it across multiple runs.

watchdog = AgentWatchdog(
    max_budget_usd=0.50,      # halt if estimated cost > $0.50
    max_identical_calls=3,    # halt if the same tool is called 3x in a row
    timeout_seconds=120,      # halt after 2 minutes regardless
    model="openai/gpt-4o",    # used only for cost estimation (no API call)
)


# ── 2. Add the callback (needs: pip install langchain) ────────────────────────
# WatchdogCallback feeds every tool call and LLM token count into the watchdog.
# It plugs into LangChain's existing callback system — no agent code changes needed.

try:
    from langchain.callbacks.base import BaseCallbackHandler  # type: ignore

    class WatchdogCallback(BaseCallbackHandler):
        """Feeds LangChain tool calls and token usage into AgentWatchdog."""

        def __init__(self, wd: AgentWatchdog) -> None:
            super().__init__()
            self.watchdog = wd

        def on_tool_start(self, serialized: dict, input_str: str, **kwargs: object) -> None:
            # Called before each tool runs. Raises WatchdogHalt if a loop is detected.
            self.watchdog.record_tool_call(serialized.get("name", "unknown"), args=input_str)

        def on_llm_end(self, response: object, **kwargs: object) -> None:
            # Called after each LLM response. Tracks token usage for budget estimation.
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            token_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            token_out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            if token_in or token_out:
                self.watchdog.record_tokens(token_in=token_in, token_out=token_out)

except ImportError:
    WatchdogCallback = None  # type: ignore[misc,assignment]


# ── 3. Run your agent safely ──────────────────────────────────────────────────

def run_agent(task: str) -> str | None:
    """
    Run a LangChain agent with watchdog protection.

    Returns the agent's answer string, or None if the watchdog halted the run.

    With a real LangChain agent, replace the mock block below with:
        result = my_executor.invoke(
            {"input": task},
            config={"callbacks": [WatchdogCallback(watchdog)]},
        )
        return result["output"]
    """
    try:
        with watchdog.watch(run_id="quickstart-run"):
            # ── Swap this block with your real AgentExecutor call ──────────────
            if WatchdogCallback is not None:
                # LangChain is installed — use the callback with a real executor:
                # result = my_executor.invoke(
                #     {"input": task},
                #     config={"callbacks": [WatchdogCallback(watchdog)]},
                # )
                # return result["output"]
                pass  # fall through to mock demo

            # ── Mock agent — simulates a loop to show the watchdog in action ──
            # Calling the same tool with the same args → watchdog halts at step 3
            for i in range(10):
                print(f"  step {i + 1}: calling search tool...")
                watchdog.record_tokens(token_in=1_000, token_out=300)
                watchdog.record_tool_call("search", args={"query": task})

            return "Done"  # only reached if no halt triggers

    except WatchdogHalt as e:
        r = e.report
        print(f"\n⚠  Watchdog halted the run:")
        print(f"   reason  → {r.reason.value}")
        print(f"   cost    → ${r.estimated_cost_usd:.4f}")
        print(f"   calls   → {len(r.tool_calls)}")

        # Suggest a fix based on the halt reason
        if r.reason == HaltReason.LOOP_DETECTED:
            print("   fix     → retry with a rephrased prompt, or increase max_identical_calls")
        elif r.reason == HaltReason.BUDGET_EXCEEDED:
            print("   fix     → raise max_budget_usd or break the task into smaller parts")
        elif r.reason == HaltReason.TIMEOUT:
            print("   fix     → raise timeout_seconds or split the task")

        return None


# ── 4. Run it ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Agent Watchdog Quick Start\n")
    answer = run_agent("What are the latest AI breakthroughs?")
    if answer:
        print(f"\nAnswer: {answer}")
    else:
        print("\nRun was halted — see above for details and suggested fix.")

    print("\nNext step: see langchain_example.py for the full integration with demos.")
