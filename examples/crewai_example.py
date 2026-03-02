"""
Agent Watchdog + CrewAI integration example.

Wraps a CrewAI crew kickoff with AgentWatchdog.
CrewAI doesn't expose per-tool callbacks easily, so we wrap at the crew level
and use the timeout + budget guards as the primary protection.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt


def run_crew_with_watchdog(crew, inputs: dict, run_id: str = "crewai-run"):
    """
    Wrap any CrewAI crew.kickoff() call with AgentWatchdog protection.

    Usage:
        from crewai import Crew, Agent, Task
        # ... define your crew ...
        result = run_crew_with_watchdog(my_crew, {"topic": "AI safety"})
    """
    watchdog = AgentWatchdog(
        max_budget_usd=1.00,      # crew runs tend to be more expensive
        max_identical_calls=5,    # crews make more legitimate repeat calls
        timeout_seconds=300,      # 5 minute hard limit
        model="anthropic/claude-sonnet-4-6",
    )

    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff(inputs=inputs)
        return result

    except WatchdogHalt as e:
        r = e.report
        print(f"\n[Watchdog] Crew halted")
        print(f"  Reason:   {r.reason.value}")
        print(f"  Elapsed:  {r.elapsed_seconds:.1f}s")
        print(f"  Cost:     ${r.estimated_cost_usd:.4f}")
        print(f"  Tools called: {len(r.tool_calls)}")
        if r.last_output:
            print(f"  Last output: {r.last_output[:200]}")
        return None


# --- Minimal CrewAI example ---
if __name__ == "__main__":
    try:
        from crewai import Agent, Task, Crew
        from crewai_tools import SerperDevTool

        researcher = Agent(
            role="Researcher",
            goal="Find concise information about the given topic",
            backstory="You are an efficient research assistant.",
            tools=[SerperDevTool()],
            verbose=False,
        )

        task = Task(
            description="Research: {topic}. Give a 3-sentence summary.",
            expected_output="A 3-sentence summary.",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[task], verbose=False)

        result = run_crew_with_watchdog(crew, {"topic": "AI agent reliability"})
        print("Result:", result)

    except ImportError:
        print("crewai not installed. This is just an example.")
        print("pip install crewai crewai-tools")
