"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with loop detection and budget guards.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# --- LangChain setup (simplified) ---
# from langchain.agents import initialize_agent, AgentType
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool

# @tool
# def search(query: str) -> str:
#     """Search the web."""
#     return f"Results for: {query}"

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# --- Watchdog integration ---

watchdog = AgentWatchdog(
    max_budget_usd=0.50,       # $0.50 max per run
    max_identical_calls=3,     # halt on 3 identical tool+args pairs
    timeout_seconds=120,       # 2 minute timeout
    model="openai/gpt-4o",
)

def run_with_watchdog(task: str, run_id: str = "langchain-run"):
    """Run a LangChain agent with watchdog protection."""
    try:
        with watchdog.watch(run_id=run_id):
            # Option A: Simple wrap (timeout + budget only, no loop detection)
            # result = agent.run(task)

            # Option B: Step-by-step with loop detection
            # for step in agent.iter(task):
            #     if "tool" in step:
            #         watchdog.record_tool_call(
            #             tool_name=step["tool"],
            #             args=step.get("tool_input"),
            #             output=step.get("observation"),
            #         )
            #     if "token_usage" in step:
            #         usage = step["token_usage"]
            #         watchdog.record_tokens(
            #             token_in=usage.get("prompt_tokens", 0),
            #             token_out=usage.get("completion_tokens", 0),
            #         )
            #
            # return step.get("output", "")

            # Placeholder for demo
            return "agent result"

    except WatchdogHalt as e:
        report = e.report
        if report.reason == HaltReason.LOOP_DETECTED:
            print(f"Agent stuck in loop: {report.message}")
        elif report.reason == HaltReason.BUDGET_EXCEEDED:
            print(f"Budget exceeded: ${report.estimated_cost_usd:.4f}")
        elif report.reason == HaltReason.TIMEOUT:
            print(f"Timed out after {report.elapsed_seconds:.1f}s")

        # Return partial result if available
        return report.last_output or "halted"


if __name__ == "__main__":
    result = run_with_watchdog("What is the capital of France?", run_id="demo-001")
    print(f"Result: {result}")
