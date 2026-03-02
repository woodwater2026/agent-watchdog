"""
Agent Watchdog + CrewAI integration example.

Wraps CrewAI crew runs with loop detection and budget guard.
Specifically addresses CrewAI issue #4495 (infinite tool-use loop regression).
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# ── CrewAI setup (standard) ──────────────────────────────────────────────────
# from crewai import Agent, Task, Crew, Process
# from crewai_tools import SerperDevTool
#
# search_tool = SerperDevTool()
# researcher = Agent(
#     role="Researcher",
#     goal="Find accurate information",
#     tools=[search_tool],
#     llm="gpt-4o",
# )
# research_task = Task(
#     description="Research {topic}",
#     agent=researcher,
#     expected_output="A summary",
# )
# crew = Crew(agents=[researcher], tasks=[research_task])

# ── Watchdog integration ─────────────────────────────────────────────────────

watchdog = AgentWatchdog(
    max_budget_usd=2.0,       # CrewAI multi-agent runs cost more
    max_identical_calls=3,    # catches the #4495 infinite loop bug
    timeout_seconds=300,
    model="openai/gpt-4o",
)


def run_crew_with_watchdog(inputs: dict, run_id: str = "crew-run"):
    """
    Wrap a CrewAI crew.kickoff() with watchdog protection.

    The watchdog catches the infinite loop regression in CrewAI 1.9.3+
    where a custom BaseTool wrapper gets called forever with empty args.
    """
    try:
        with watchdog.watch(run_id=run_id):
            # result = crew.kickoff(inputs=inputs)
            result = _simulate_looping_crew(inputs)
            return result

    except WatchdogHalt as e:
        report = e.report
        print(f"\n🛑 Crew halted: {report.reason.value}")
        print(f"   Run: {report.run_id}")
        print(f"   Elapsed: {report.elapsed_seconds:.1f}s")
        print(f"   Cost: ${report.estimated_cost_usd:.4f}")
        print(f"   Tool calls made: {len(report.tool_calls)}")
        if report.reason.value == "loop_detected":
            print("   ⚠️  Possible infinite tool loop — check tool wrapper implementation")
        return {"halted": True, "reason": report.reason.value}


def _simulate_looping_crew(inputs: dict):
    """Simulates CrewAI issue #4495: tool called with empty args in a loop."""
    broken_tool_calls = [
        ("BedrockKBRetrieverTool", ""),   # empty args — the regression
        ("BedrockKBRetrieverTool", ""),
        ("BedrockKBRetrieverTool", ""),   # watchdog fires here
    ]
    for tool_name, args in broken_tool_calls:
        watchdog.record_tool_call(tool_name, args=args)
        watchdog.record_tokens(token_in=300, token_out=50)
    return {"output": "unreachable"}


if __name__ == "__main__":
    print("Running CrewAI crew with watchdog protection...")
    result = run_crew_with_watchdog({"topic": "AI agent frameworks"}, run_id="crew-001")
    print(f"Result: {result}")
