import sys

from langchain_core.messages import HumanMessage

from travel_agent.agent import build_agent_executor


def main() -> None:
    executor = build_agent_executor()
    thread_config = {"configurable": {"thread_id": "cli-session"}}

    sys.stdout.write("Travel Agent — type 'exit' or 'quit' to leave.\n")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}, thread_config
        )
        sys.stdout.write(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
