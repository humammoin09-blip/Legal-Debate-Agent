"""
prompts.py - System Prompts and formatting utilities for AI Devil's Advocate.

Kept modular so prompt engineering, style guidelines, or new perspectives can
be tuned independently of the graph orchestration logic.
"""

AGENT_A_SYSTEM_PROMPT = """You are a sharp debater arguing FOR the given topic.
Be persuasive, use logic and evidence-style reasoning, and directly rebut the opponent's last point if one exists.
Keep response under 120 words. Focus on strong arguments and crisp delivery."""

AGENT_B_SYSTEM_PROMPT = """You are a sharp debater arguing AGAINST the given topic.
Be persuasive, use logic and evidence-style reasoning, and directly rebut the opponent's last point if one exists.
Keep response under 120 words. Focus on strong arguments and crisp delivery."""

JUDGE_SYSTEM_PROMPT = """You are a neutral judge. Read the full debate transcript and give a balanced verdict — summarize the strongest point from each side, note any weak/fallacious arguments, and conclude with a nuanced final take. Do NOT declare one side the outright winner unless the argument quality is genuinely lopsided."""


def format_transcript_for_prompt(transcript: list[dict[str, str]]) -> str:
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


def get_agent_prompt(role: str, topic: str, transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Builds the chat messages list for Agent A, Agent B, or the Judge.
    """
    if role == "Agent A":
        system_content = AGENT_A_SYSTEM_PROMPT
        user_content = (
            f"Topic: {topic}\n\n"
            f"Debate Transcript So Far:\n{format_transcript_for_prompt(transcript)}\n\n"
            f"Your stance: IN FAVOR of '{topic}'. Provide your argument (under 120 words):"
        )
    elif role == "Agent B":
        system_content = AGENT_B_SYSTEM_PROMPT
        user_content = (
            f"Topic: {topic}\n\n"
            f"Debate Transcript So Far:\n{format_transcript_for_prompt(transcript)}\n\n"
            f"Your stance: AGAINST '{topic}'. Provide your argument (under 120 words):"
        )
    elif role == "Judge":
        system_content = JUDGE_SYSTEM_PROMPT
        user_content = (
            f"Topic: {topic}\n\n"
            f"Complete Debate Transcript:\n{format_transcript_for_prompt(transcript)}\n\n"
            f"Provide your structured verdict and balanced assessment:"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
