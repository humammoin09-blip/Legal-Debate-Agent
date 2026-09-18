"""
prompts.py - System Prompts and formatting utilities for AI Devil's Advocate.

Kept modular so prompt engineering, style guidelines, personas, or new perspectives can
be tuned independently of the graph orchestration logic.
"""

from typing import List, Dict, Optional

DEBATE_PERSONAS: Dict[str, str] = {
    "Analytical & Balanced": "Debate Persona [Analytical & Balanced]: Rigorous, structured, analytical, data-driven, and objective.",
    "Aggressive / Prosecutor Style": "Debate Persona [Aggressive / Prosecutor Style]: High-intensity, rhetorical, cross-examining, assertive, and relentless in exposing logical fallacies.",
    "Philosophical & Deep": "Debate Persona [Philosophical & Deep]: Socratic, philosophical, thought-provoking, addressing first principles, existential/ethical ramifications, and deep abstract concepts.",
    "Humorous & Sarcastic": "Debate Persona [Humorous & Sarcastic]: Witty, sharp, slightly satirical, punchy, and entertaining while maintaining logical substance.",
}

AGENT_A_SYSTEM_PROMPT = """You are a sharp debater arguing FOR the given topic.
Be persuasive, use logic and evidence-style reasoning, and directly rebut the opponent's last point or any audience cross-examination if present.
Keep response under 120 words. Focus on strong arguments and crisp delivery."""

AGENT_B_SYSTEM_PROMPT = """You are a sharp debater arguing AGAINST the given topic.
Be persuasive, use logic and evidence-style reasoning, and directly rebut the opponent's last point or any audience cross-examination if present.
Keep response under 120 words. Focus on strong arguments and crisp delivery."""

JUDGE_SYSTEM_PROMPT = """You are a neutral judge. Read the full debate transcript and give a balanced verdict — summarize the strongest point from each side, note any weak/fallacious arguments, address any audience cross-examination points raised, and conclude with a nuanced final take. Do NOT declare one side the outright winner unless the argument quality is genuinely lopsided."""


def format_transcript_for_prompt(transcript: List[Dict[str, str]]) -> str:
    """
    Formats the list of speaker-text dictionary turns into a readable text history.
    """
    if not transcript:
        return "(No arguments made yet. You are giving the opening statement.)"

    formatted_lines = []
    for turn in transcript:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        formatted_lines.append(f"[{speaker}]:\n{text}\n")
    return "\n".join(formatted_lines)


def get_agent_prompt(
    role: str,
    topic: str,
    transcript: List[Dict[str, str]],
    persona: str = "Analytical & Balanced",
    user_intervention: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Builds the chat messages list for Agent A, Agent B, or the Judge.
    Incorporates the chosen persona style and any active cross-examination point.
    """
    persona_guideline = DEBATE_PERSONAS.get(persona, DEBATE_PERSONAS["Analytical & Balanced"])
    intervention_context = ""
    if user_intervention and user_intervention.strip():
        intervention_context = f"\n[Audience Cross-Examination / Interjection]:\n\"{user_intervention.strip()}\"\n(You must directly address this counter-point in your argument.)\n"

    if role == "Agent A":
        system_content = f"{AGENT_A_SYSTEM_PROMPT}\n{persona_guideline}"
        user_content = (
            f"Topic: {topic}\n\n"
            f"Debate Transcript So Far:\n{format_transcript_for_prompt(transcript)}\n"
            f"{intervention_context}\n"
            f"Your stance: IN FAVOR of '{topic}'. Provide your argument (under 120 words):"
        )
    elif role == "Agent B":
        system_content = f"{AGENT_B_SYSTEM_PROMPT}\n{persona_guideline}"
        user_content = (
            f"Topic: {topic}\n\n"
            f"Debate Transcript So Far:\n{format_transcript_for_prompt(transcript)}\n"
            f"{intervention_context}\n"
            f"Your stance: AGAINST '{topic}'. Provide your argument (under 120 words):"
        )
    elif role == "Judge":
        system_content = JUDGE_SYSTEM_PROMPT
        user_content = (
            f"Topic: {topic}\n\n"
            f"Complete Debate Transcript:\n{format_transcript_for_prompt(transcript)}\n"
            f"{intervention_context}\n"
            f"Provide your structured verdict and balanced assessment:"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
