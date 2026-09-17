"""
app.py - Streamlit Frontend for AI Devil's Advocate.

Features:
- Multi-Provider Support (Groq, Google Gemini, OpenRouter)
- Automatic 429 Rate Limit Fallback
- Theme Switcher (Dark / Light / System) with dynamically adapted CSS
- Clean modern layout with Inter/Poppins fonts and zero trailing whitespace scroll
- Verdict highlight card with distinct styling
- Export to Markdown and formatted PDF
- Live Session API Call Counter
- Safe Sidebar State initialization and toggling
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment
load_dotenv()

from llm_factory import (
    PROVIDER_MODELS,
    PROVIDER_DEFAULT_MODELS,
    get_default_key_for_provider,
    FallbackLLMWrapper,
)
from debate_graph import build_debate_graph
from export_utils import generate_markdown_transcript, generate_pdf_transcript

# Page configuration
st.set_page_config(
    page_title="AI Devil's Advocate | Multi-Agent Debate",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "debate_completed" not in st.session_state:
    st.session_state.debate_completed = False

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "api_calls_count" not in st.session_state:
    st.session_state.api_calls_count = 0

if "fallback_notifications" not in st.session_state:
    st.session_state.fallback_notifications = []

if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "System"


def reset_debate():
    """Resets the debate state for a fresh run."""
    st.session_state.transcript = []
    st.session_state.debate_completed = False
    st.session_state.is_running = False
    st.session_state.topic = ""
    st.session_state.fallback_notifications = []


def increment_call_counter():
    """Increments the session API call counter."""
    st.session_state.api_calls_count += 1


# ----------------- DYNAMIC THEME CSS GENERATOR -----------------
def get_theme_css(theme: str) -> str:
    """Generates scoped CSS for Dark, Light, or System themes."""
    base_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Constrain main container to eliminate infinite scroll & center layout */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        margin-left: auto;
        margin-right: auto;
    }

    /* Transparent header to preserve sidebar toggle button accessibility */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    #MainMenu { visibility: hidden; }
    footer { display: none !important; }

    /* Animated message cards with smooth entrance */
    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .agent-card {
        animation: cardFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .verdict-card {
        animation: cardFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        border-radius: 12px;
        padding: 22px 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .verdict-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .turn-text, .verdict-body {
        font-size: 0.97rem;
        line-height: 1.65;
    }

    .speaker-badge-a, .speaker-badge-b {
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .speaker-badge-a { color: #2563EB; }
    .speaker-badge-b { color: #DC2626; }

    /* Animated thinking indicator */
    .thinking-box {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        margin-top: 6px;
        margin-bottom: 14px;
    }

    .typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .dot {
        width: 6px;
        height: 6px;
        background-color: #3B82F6;
        border-radius: 50%;
        animation: pulseDot 1.4s infinite ease-in-out both;
    }

    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes pulseDot {
        0%, 80%, 100% {
            transform: scale(0.4);
            opacity: 0.4;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }

    .usage-badge {
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 500;
        font-size: 0.88rem;
        margin-bottom: 15px;
    }

    .stChatMessage {
        background: transparent !important;
        padding: 0 !important;
    }
    """

    light_theme_rules = """
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        color: #1E293B;
    }
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-bottom: 1.75rem;
    }
    .agent-a-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
    }
    .agent-b-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #DC2626;
    }
    .turn-text {
        color: #334155;
    }
    .verdict-card {
        background: #F8FAFC;
        border: 1.5px solid #6366F1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
    }
    .verdict-header {
        color: #4F46E5;
    }
    .verdict-body {
        color: #1E293B;
    }
    .thinking-box {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #475569;
    }
    .usage-badge {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
    }
    """

    dark_theme_rules = """
    .stApp {
        background: linear-gradient(180deg, #0B0F19 0%, #111827 100%) !important;
        color: #F8FAFC !important;
    }
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC !important;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #94A3B8 !important;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-bottom: 1.75rem;
    }
    .agent-a-box {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #3B82F6 !important;
    }
    .agent-b-box {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #EF4444 !important;
    }
    .speaker-badge-a { color: #60A5FA !important; }
    .speaker-badge-b { color: #F87171 !important; }
    .turn-text {
        color: #F1F5F9 !important;
    }
    .verdict-card {
        background: #1E1B4B !important;
        border: 1.5px solid #818CF8 !important;
        box-shadow: 0 4px 16px rgba(129, 140, 248, 0.15) !important;
    }
    .verdict-header {
        color: #C7D2FE !important;
    }
    .verdict-body {
        color: #E0E7FF !important;
    }
    .thinking-box {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #94A3B8 !important;
    }
    .usage-badge {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
    }
    """

    if theme == "Dark":
        return base_css + dark_theme_rules + "</style>"
    elif theme == "Light":
        return base_css + light_theme_rules + "</style>"
    else:  # System
        return (
            base_css
            + light_theme_rules
            + "@media (prefers-color-scheme: dark) {\n"
            + dark_theme_rules
            + "\n}\n</style>"
        )


# Apply dynamic theme CSS
st.markdown(get_theme_css(st.session_state.selected_theme), unsafe_allow_html=True)


# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.title("Debate Settings")

    # Theme Switcher (Dark / Light / System)
    theme_options = ["System", "Dark", "Light"]
    current_theme_idx = theme_options.index(st.session_state.selected_theme) if st.session_state.selected_theme in theme_options else 0
    selected_theme = st.selectbox(
        "Theme",
        options=theme_options,
        index=current_theme_idx,
        help="Switch between Dark, Light, or System default appearance.",
    )
    if selected_theme != st.session_state.selected_theme:
        st.session_state.selected_theme = selected_theme
        st.rerun()

    # Live Call Counter
    st.markdown(
        f'<div class="usage-badge">API Calls this session: <b>{st.session_state.api_calls_count}</b></div>',
        unsafe_allow_html=True,
    )

    st.subheader("LLM Provider & Model")

    # 1. Provider Dropdown
    provider_options = ["Groq", "Google Gemini", "OpenRouter"]
    selected_provider = st.selectbox(
        "Active Provider",
        options=provider_options,
        index=0,
        help="Choose the primary LLM provider for the debate.",
    )

    # 2. Model Dropdown based on active provider
    available_models = PROVIDER_MODELS.get(selected_provider, [])
    default_model_for_prov = PROVIDER_DEFAULT_MODELS.get(selected_provider, available_models[0])
    default_idx = available_models.index(default_model_for_prov) if default_model_for_prov in available_models else 0

    selected_model = st.selectbox(
        f"Model ({selected_provider})",
        options=available_models,
        index=default_idx,
    )

    # 3. Primary Provider API Key Input
    def get_effective_key(prov_name: str) -> str:
        env_key = get_default_key_for_provider(prov_name)
        sec_key = (
            st.secrets.get(f"{prov_name.upper().replace(' ', '_')}_API_KEY", "")
            if hasattr(st, "secrets")
            else ""
        )
        return env_key or sec_key

    primary_default_key = get_effective_key(selected_provider)
    primary_api_key = st.text_input(
        f"{selected_provider} API Key",
        value=primary_default_key,
        type="password",
        help=f"Enter API key for {selected_provider}.",
    )

    # Secondary API keys expander for automatic 429 fallback
    with st.expander("Fallback Providers (Auto-Retry on 429)", expanded=False):
        st.caption("Add secondary keys. If your active provider hits a 429 rate limit, the debate will continue with the next available provider.")
        fallback_keys = {}
        for p in provider_options:
            if p != selected_provider:
                def_key = get_effective_key(p)
                k_val = st.text_input(f"{p} Key", value=def_key, type="password", key=f"fallback_key_{p}")
                if k_val.strip():
                    fallback_keys[p] = k_val.strip()

    st.markdown("---")

    # Number of Debate Rounds Slider (2-5)
    max_rounds = st.slider(
        "Number of Debate Rounds",
        min_value=2,
        max_value=5,
        value=3,
        help="Each round features 1 argument from Agent A and 1 rebuttal from Agent B.",
    )

    st.markdown("---")
    st.subheader("Example Topics")
    sample_topics = [
        "Artificial Intelligence should be granted legal personhood.",
        "Remote work is permanently superior to office work.",
        "Social media platforms should ban anonymous accounts.",
        "Universal Basic Income is essential in an automated world.",
        "Open-source AI models pose greater risks than proprietary ones.",
    ]

    for sample in sample_topics:
        if st.button(sample, key=f"sample_{sample[:18]}", use_container_width=True):
            st.session_state.topic = sample
            st.session_state.transcript = []
            st.session_state.debate_completed = False
            st.session_state.fallback_notifications = []

    st.markdown("---")
    if st.button("Reset / New Debate", use_container_width=True, type="secondary"):
        reset_debate()
        st.rerun()


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">AI Devil\'s Advocate</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Autonomous multi-agent debate engine powered by LangGraph. Two opposing AI agents challenge each other across multiple rounds, concluding with a neutral judicial verdict.</div>',
    unsafe_allow_html=True,
)

# Render Fallback notices if any occurred
if st.session_state.fallback_notifications:
    for notice in st.session_state.fallback_notifications:
        st.warning(notice)

# Topic input section
col_input, col_btn = st.columns([5, 1.2])

with col_input:
    topic_input = st.text_input(
        "Debate Topic or Statement",
        value=st.session_state.topic,
        placeholder="Enter a topic or statement to debate...",
        label_visibility="collapsed",
    )

with col_btn:
    start_btn = st.button("Start Debate", type="primary", use_container_width=True)

if topic_input:
    st.session_state.topic = topic_input


# ----------------- CHAT CONTAINER & RENDER FUNCTIONS -----------------
chat_container = st.container()

def render_turn(speaker: str, text: str):
    """Renders a single turn with custom speaker styles."""
    if speaker == "Agent A":
        st.markdown(
            f'<div class="agent-card agent-a-box">'
            f'<div class="speaker-badge-a">Agent A (Arguing FOR)</div>'
            f'<div class="turn-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif speaker == "Agent B":
        st.markdown(
            f'<div class="agent-card agent-b-box">'
            f'<div class="speaker-badge-b">Agent B (Arguing AGAINST)</div>'
            f'<div class="turn-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif speaker == "Judge":
        st.markdown(
            f'<div class="verdict-card">'
            f'<div class="verdict-header">Neutral Judge\'s Verdict & Final Assessment</div>'
            f'<div class="verdict-body">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


with chat_container:
    for turn in st.session_state.transcript:
        render_turn(turn["speaker"], turn["text"])


# ----------------- RUN DEBATE ORCHESTRATION -----------------
if start_btn:
    # 1. Validation: Topic
    if not st.session_state.topic.strip():
        st.warning("Please enter a debate topic or pick an example from the sidebar.")
        st.stop()

    # 2. Validation: Active Provider API Key
    eff_primary_key = primary_api_key.strip()
    if not eff_primary_key:
        st.error(f"{selected_provider} API Key is missing. Please enter it in the sidebar.")
        st.stop()

    # Reset transcript for new run
    st.session_state.transcript = []
    st.session_state.debate_completed = False
    st.session_state.is_running = True
    st.session_state.fallback_notifications = []

    # Build provider configs chain (Primary -> Fallbacks)
    providers_chain = [
        {
            "provider": selected_provider,
            "api_key": eff_primary_key,
            "model_name": selected_model,
        }
    ]

    for fb_provider, fb_key in fallback_keys.items():
        providers_chain.append(
            {
                "provider": fb_provider,
                "api_key": fb_key,
                "model_name": PROVIDER_DEFAULT_MODELS.get(fb_provider),
            }
        )

    # Fallback handler callback
    def on_fallback_triggered(from_p: str, to_p: str, err_desc: str):
        msg = f"Switched to {to_p} from {from_p} due to rate limit (429)."
        st.session_state.fallback_notifications.append(msg)
        st.toast(msg, icon="🔄")

    # Instantiate the resilient FallbackLLMWrapper
    llm_wrapper = FallbackLLMWrapper(
        providers_configs=providers_chain,
        on_fallback=on_fallback_triggered,
        on_call_completed=increment_call_counter,
    )

    try:
        # Build LangGraph workflow with provider-agnostic wrapper
        app_graph = build_debate_graph(llm=llm_wrapper)

        initial_state = {
            "topic": st.session_state.topic.strip(),
            "transcript": [],
            "round_count": 0,
            "max_rounds": max_rounds,
        }

        status_placeholder = st.empty()

        # Stream node execution
        with chat_container:
            for event in app_graph.stream(initial_state, stream_mode="updates"):
                for node_name, state_update in event.items():
                    if "transcript" in state_update:
                        new_turns = state_update["transcript"]
                        if new_turns:
                            latest_turn = new_turns[-1]
                            st.session_state.transcript.append(latest_turn)
                            render_turn(latest_turn["speaker"], latest_turn["text"])

                    # Animated thinking status
                    if node_name == "agent_a":
                        status_placeholder.markdown(
                            """
                            <div class="thinking-box">
                                <span>Agent B is formulating a rebuttal</span>
                                <div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif node_name == "agent_b":
                        current_rounds = state_update.get("round_count", 0)
                        if current_rounds < max_rounds:
                            status_placeholder.markdown(
                                f"""
                                <div class="thinking-box">
                                    <span>Round {current_rounds}/{max_rounds} complete. Agent A is preparing next argument</span>
                                    <div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            status_placeholder.markdown(
                                """
                                <div class="thinking-box">
                                    <span>Max rounds reached. The Judge is evaluating all arguments</span>
                                    <div class="typing-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    elif node_name == "judge":
                        status_placeholder.empty()

        st.session_state.debate_completed = True
        st.session_state.is_running = False
        st.rerun()

    except Exception as e:
        error_msg = str(e)
        st.session_state.is_running = False
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            st.error(
                "Rate Limit Exceeded across all configured providers: "
                "Please wait 30-60 seconds or configure additional fallback provider keys in the sidebar."
            )
        elif "api_key" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
            st.error(f"Invalid API Key for {selected_provider}: Please verify your key in the sidebar.")
        else:
            st.error(f"An unexpected error occurred: {error_msg}")


# ----------------- EXPORT & DOWNLOAD SECTION -----------------
if st.session_state.debate_completed and st.session_state.transcript:
    st.markdown("---")
    st.subheader("Export & Download Transcript")

    col_md, col_pdf, _ = st.columns([1.5, 1.5, 3])

    # 1. Markdown Export
    md_content = generate_markdown_transcript(st.session_state.topic, st.session_state.transcript)
    safe_topic_slug = "".join(c if c.isalnum() else "_" for c in st.session_state.topic[:30])

    with col_md:
        st.download_button(
            label="Download Markdown (.md)",
            data=md_content,
            file_name=f"debate_{safe_topic_slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # 2. PDF Export
    try:
        pdf_bytes = generate_pdf_transcript(st.session_state.topic, st.session_state.transcript)
        with col_pdf:
            st.download_button(
                label="Download PDF (.pdf)",
                data=pdf_bytes,
                file_name=f"debate_{safe_topic_slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    except Exception as pdf_err:
        with col_pdf:
            st.caption(f"PDF generation note: {pdf_err}")
