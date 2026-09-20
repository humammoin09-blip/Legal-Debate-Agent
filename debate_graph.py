"""
debate_graph.py - LangGraph orchestration for AI Devil's Advocate.

Defines the multi-agent state schema, debater nodes, judge node,
and conditional routing logic. Accepts provider-agnostic LLM clients,
persona styles, and audience cross-examination interjections.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from prompts import get_agent_prompt, DEBATE_PERSONAS
from llm_factory import create_llm, FallbackLLMWrapper, PROVIDER_DEFAULT_MODELS, extract_text_content


class Turn(TypedDict):
    speaker: str  # "Agent A" | "Agent B" | "Judge" | "Audience"
    text: str


class DebateState(TypedDict, total=False):
    topic: str
    transcript: List[Dict[str, str]]
    round_count: int
    max_rounds: int
    persona: str
    user_intervention: Optional[str]


def build_debate_graph(
    llm: Optional[Any] = None,
    api_key: Optional[str] = None,
    provider: str = "Groq",
    model_name: Optional[str] = None,
):
    """
    Constructs and compiles the LangGraph StateGraph for the multi-agent debate.
    Accepts any LLM instance (e.g., BaseChatModel, FallbackLLMWrapper) or builds one
    from the provider/api_key params.
    """
    if llm is None:
        if not api_key:
            raise ValueError(f"API key is required for provider '{provider}'.")
        effective_model = model_name or PROVIDER_DEFAULT_MODELS.get(provider)
        llm = create_llm(provider=provider, api_key=api_key, model_name=effective_model)

    def agent_a_node(state: DebateState) -> Dict[str, Any]:
        """
        Agent A node: Argues FOR the topic.
        Reads the full transcript so far and generates the next argument.
        """
        persona = state.get("persona", "Analytical & Balanced")
        user_intervention = state.get("user_intervention", None)
        messages_spec = get_agent_prompt(
            "Agent A",
            state["topic"],
            state["transcript"],
            persona=persona,
            user_intervention=user_intervention,
        )
        lc_messages = [
            SystemMessage(content=messages_spec[0]["content"]),
            HumanMessage(content=messages_spec[1]["content"]),
        ]
        response = llm.invoke(lc_messages)
        new_turn = {"speaker": "Agent A", "text": extract_text_content(response)}
        return {
            "transcript": state["transcript"] + [new_turn]
        }

    def agent_b_node(state: DebateState) -> Dict[str, Any]:
        """
        Agent B node: Argues AGAINST the topic.
        Reads the full transcript (including Agent A's latest response) and rebuts.
        Increments the round_count.
        """
        persona = state.get("persona", "Analytical & Balanced")
        user_intervention = state.get("user_intervention", None)
        messages_spec = get_agent_prompt(
            "Agent B",
            state["topic"],
            state["transcript"],
            persona=persona,
            user_intervention=user_intervention,
        )
        lc_messages = [
            SystemMessage(content=messages_spec[0]["content"]),
            HumanMessage(content=messages_spec[1]["content"]),
        ]
        response = llm.invoke(lc_messages)
        new_turn = {"speaker": "Agent B", "text": extract_text_content(response)}
        return {
            "transcript": state["transcript"] + [new_turn],
            "round_count": state["round_count"] + 1,
        }

    def judge_node(state: DebateState) -> Dict[str, Any]:
        """
        Judge node: Evaluates the complete debate after max rounds are reached.
        Provides a neutral, structured verdict.
        """
        persona = state.get("persona", "Analytical & Balanced")
        user_intervention = state.get("user_intervention", None)
        messages_spec = get_agent_prompt(
            "Judge",
            state["topic"],
            state["transcript"],
            persona=persona,
            user_intervention=user_intervention,
        )
        lc_messages = [
            SystemMessage(content=messages_spec[0]["content"]),
            HumanMessage(content=messages_spec[1]["content"]),
        ]
        response = llm.invoke(lc_messages)
        new_turn = {"speaker": "Judge", "text": extract_text_content(response)}
        return {
            "transcript": state["transcript"] + [new_turn]
        }

    def should_continue(state: DebateState) -> str:
        """
        Conditional router after Agent B finishes:
        - If round_count < max_rounds: loop back to Agent A.
        - Otherwise: route to Judge.
        """
        if state["round_count"] < state["max_rounds"]:
            return "agent_a"
        return "judge"

    # Define Graph
    workflow = StateGraph(DebateState)

    workflow.add_node("agent_a", agent_a_node)
    workflow.add_node("agent_b", agent_b_node)
    workflow.add_node("judge", judge_node)

    # Add edges
    workflow.add_edge(START, "agent_a")
    workflow.add_edge("agent_a", "agent_b")
    workflow.add_conditional_edges(
        "agent_b",
        should_continue,
        {
            "agent_a": "agent_a",
            "judge": "judge",
        },
    )
    workflow.add_edge("judge", END)

    return workflow.compile()
