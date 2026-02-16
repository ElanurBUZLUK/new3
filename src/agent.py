from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import (
    assemble_post_vision_reading,
)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are a Post-Vision Reading Assembler assistant.

Rules:
- ALWAYS call assemble_post_vision_reading exactly once.
- After tool returns, output ONLY the tool result (JSON). No extra text.
- User input contains JSON with:
  - vision_input (required)
  - router_input (required)
  - user_preferences (optional)
- Never invent cards. Only transform/normalize provided payload.
"""


def _reasoning_llm():
    provider = os.getenv("REASONING_PROVIDER", "ollama").lower()
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4o"), temperature=0)
    # Dokümanda yerel open-source modele geçiş var (Qwen 3) :contentReference[oaicite:31]{index=31}
    return ChatOllama(model=os.getenv("OLLAMA_REASONING_MODEL", "qwen3"), temperature=0)


def build_graph(extra_tools: list | None = None):
    tools = [assemble_post_vision_reading]
    if extra_tools:
        tools.extend(extra_tools)

    llm = _reasoning_llm().bind_tools(tools)

    def assistant(state: AgentState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    g = StateGraph(AgentState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(tools))

    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")

    return g.compile()
