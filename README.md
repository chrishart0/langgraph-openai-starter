# PC Builder Assistant

A LangGraph-based AI assistant that helps users build PCs based on their experience level. The assistant adapts its language and recommendations to match the user's expertise, from beginners to experienced builders.

## Setup

1. Install dependencies using uv:
```bash
uv sync
```

2. Create a `.env` file in the root directory with the following template:
```bash
LANGSMITH_PROJECT=my-project
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_

OPENAI_API_KEY=sk-proj-
```

Replace the placeholder values with your actual API keys:
- `LANGSMITH_API_KEY`: Your LangSmith API key
- `OPENAI_API_KEY`: Your OpenAI API key

## Running the Assistant

Start the development server:
```bash
uv run langgraph dev
```

This will start the LangGraph development server, which you can interact with through the provided interface.

## Experience Levels

The assistant supports three experience levels:
- `beginner`: For first-time builders
- `intermediate`: For users with some experience
- `pro`: For experienced builders

To specify an experience level, use the `experience` parameter in the configuration:
```python
config={"configurable": {"experience": "beginner"}}
```

## Example Usage

```python
from langchain_core.messages import HumanMessage
import asyncio
from src.agent.graph import graph

async def main():
    response = await graph.ainvoke(
        {"messages": [HumanMessage(content="I need help building a gaming PC with a $1000 budget")]},
        config={"configurable": {"experience": "intermediate"}},
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```
