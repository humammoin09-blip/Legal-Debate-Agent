#  AI Devil's Advocate — Multi-Agent Debate Application

An autonomous multi-agent debate application built with **Python**, **LangGraph**, and **Streamlit**. Supports multiple free-tier and cloud LLM providers (**Groq**, **Google Gemini**, **OpenRouter**) with automatic 429 rate-limit fallback, polished animated UI, and formatted PDF/Markdown exports.

---

##  Key Features

1. **Multi-Provider Architecture:**
   - **Groq:** Ultra-low latency (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, etc.)
   - **Google Gemini:** Google GenAI (`gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`)
   - **OpenRouter:** Unified API (`meta-llama/llama-3.1-8b-instruct:free`, `google/gemini-2.0-flash-exp:free`, etc.)
2. ** Automatic Rate-Limit (429) Fallback:**
   - If your active provider hits a rate limit or quota exhaustion, the app automatically switches to your next configured provider on the fly with zero interruption to the ongoing debate.
3. ** Polished UI & Animations:**
   - Google Fonts (Inter / Poppins), smooth message fade-in animations, bouncing typing/thinking dots, and custom stylized debater & verdict cards.
4. ** Transcript Exports:**
   - 1-click downloads for styled **PDF** (`.pdf`) and formatted **Markdown** (`.md`).
5. ** Session API Call Counter:**
   - Real-time counter in the sidebar to monitor your free-tier usage.

---

## 📁 Project Structure

```text
├── app.py               # Streamlit frontend with animated chat UI, provider switcher, & exports
├── debate_graph.py      # LangGraph state machine with provider-agnostic LLM orchestration
├── llm_factory.py       # Factory for Groq, Gemini, OpenRouter + FallbackLLMWrapper
├── export_utils.py      # PDF (ReportLab) and Markdown export generators
├── prompts.py           # Modular system prompts for Agent A, Agent B, and the Judge
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables (GROQ, GEMINI, OPENROUTER)
└── README.md            # Documentation & deployment guide
```

---

##  Quick Start (Local Setup)

### 1. Clone or Open the Repository
```bash
cd "Legal Debate Agent"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and fill in at least one provider key:
```bash
cp .env.example .env
```

```env
# Free keys:
GROQ_API_KEY=gsk_...               # https://console.groq.com/keys
GEMINI_API_KEY=...                  # https://aistudio.google.com/app/apikey
OPENROUTER_API_KEY=sk-or-v1-...     # https://openrouter.ai/keys
```
*(Note: You can also enter or change API keys directly inside the Streamlit sidebar at any time!)*

### 4. Run the App
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

1. Push your repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and create a **New App**.
3. Under **Advanced Settings** -> **Secrets**, add your provider keys:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GEMINI_API_KEY = "..."
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
