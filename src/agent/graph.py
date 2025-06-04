"""LangGraph PC Building Assistant with typed prompts in code."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph

# 1. Make Configuration expect a 'experience' that can be "beginner", "intermediate", or "pro"
class Configuration(TypedDict):
    experience: Literal["beginner", "intermediate", "pro"]


# 2. Initialize your LLM as usual
llm = init_chat_model("openai:gpt-4.1-nano", temperature=0.7)

# 3. Define a helper that maps experience → actual system prompt text
def get_system_prompt(experience: str) -> str:
    if experience == "beginner":
        return (
            "You are PCBuilderAI, a helpful PC building assistant for beginners. "
            "Explain concepts clearly and avoid technical jargon. "
            "Focus on pre-built options and simple upgrades. "
            "Always explain why you're suggesting specific parts."
        )
    elif experience == "intermediate":
        return (
            "You are PCBuilderAI, a helpful PC building assistant for users with some experience. "
            "You can use technical terms but explain them briefly. "
            "Focus on balanced builds with room for future upgrades. "
            "Include some advanced features but keep them optional."
        )
    # pro mode
    return (
        "You are PCBuilderAI, a helpful PC building assistant for experienced builders. "
        "You can use technical terms freely and discuss advanced concepts. "
        "Focus on performance optimization, overclocking potential, and advanced features. "
        "Include detailed specifications and technical considerations."
    )


async def pc_builder_node(state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
    """
    - Pull 'experience' from config.
    - Translate it into a SystemMessage via get_system_prompt().
    - Prepend that SystemMessage (once) before sending to the LLM.
    - Return the LLM response appended to the message history.
    """
    messages = state["messages"]
    experience = config["configurable"].get("experience", "intermediate")
    system_text = get_system_prompt(experience)

    # If there isn't already a SystemMessage at index 0, insert it now
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_text)] + messages

    # Invoke LLM
    response = await llm.ainvoke(messages)  # returns an AIMessage

    # Append to state
    return {"messages": messages + [response]}


# 4. Build the graph, telling LangGraph about our Configuration schema
graph = (
    StateGraph(MessagesState, config_schema=Configuration)
    .add_node("pc_builder_node", pc_builder_node)
    .add_edge("__start__", "pc_builder_node")
    .compile(name="PCBuilderAI Graph")
)


if __name__ == "__main__":
    # Example local test (you wouldn't normally run this directly in production)
    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="I need help building a gaming PC with a $1000 budget")]},
            config={"configurable": {"experience": "intermediate"}},
        )
    )
