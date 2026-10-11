"""
Princess Yacht VR Experience — Streamlit Edition
Powered by Groq & Streamlit Community Cloud. Requires Python 3.10+.
"""

import os
import json
import traceback
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
import httpx
from openai import OpenAI, APIError

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIG (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Princess Yacht VR",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 2. CONSTANTS / VR CONFIG
# ─────────────────────────────────────────────────────────────
YACHT_MODELS = {
    "Princess Y85": {
        "Compact": "https://vrcloud.com/?pv=1550754573&play=1",
        "Tall":    "https://vrcloud.com/?pv=1550754573&play=1",
        "Full":    "https://vrcloud.com/?pv=1550754573&play=1",
    },
    "Princess X95": {
        "Compact": "https://vrcloud.com/?pv=1620886782&play=1",
        "Tall":    "https://vrcloud.com/?pv=1620886782&play=1",
        "Full":    "https://vrcloud.com/?pv=1620886782&play=1",
    },
}

LLM_MODELS = {
    "🧠 GPT-OSS 20B (fast reasoning)": "openai/gpt-oss-20b",
    "🧠 GPT-OSS 120B (stronger reasoning)": "openai/gpt-oss-120b",
    "⚡ Qwen 3.6 27B": "qwen/qwen3.6-27b",
}
DEFAULT_LLM_LABEL = "🧠 GPT-OSS 20B (fast reasoning)"

DEFAULT_VR_HTML = """
<div style="text-align:center; padding: 24px; border: 2px dashed #cfd8dc;
            border-radius: 12px; background: #f8fafc; color:#37474f;">
    <h3>🚢 Ask the chat to start the Virtual Reality Experience!</h3>
    <p>Try: <b>Show me the X95 full VR view</b></p>
    <p style="font-size:13px; color:#607d8b;">
        Or use the sidebar to pick a yacht &amp; view directly.
    </p>
</div>
"""

SYSTEM_PROMPT = (
    "You are a helpful yacht expert specializing in Princess Yachts, "
    "with deep knowledge of the X95 and Y85 models. Keep answers concise, "
    "enthusiastic, and factual."
)

SUGGESTED_PROMPTS = [
    "Tell me about the Princess X95",
    "Show me the X95 full VR view",
    "Compare X95 vs Y85",
    "What's the top speed of the Y85?",
]

# ─────────────────────────────────────────────────────────────
# 3. CLIENT INITIALIZATION
# ─────────────────────────────────────────────────────────────
def _get_groq_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = _get_groq_key()
    if not key:
        return None, "GROQ_API_KEY is missing (check Streamlit Secrets or environment variables)."
    try:
        http_client = httpx.Client(timeout=60.0)
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
            http_client=http_client,
        )
        return client, None
    except Exception as e:
        return None, f"Client init failed: {e}"

groq_client, client_err = get_groq_client()

# ─────────────────────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vr_url" not in st.session_state:
    st.session_state.vr_url = None
if "vr_model" not in st.session_state:
    st.session_state.vr_model = "Princess Y85"
if "vr_view" not in st.session_state:
    st.session_state.vr_view = "Full"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "llm_label" not in st.session_state:
    st.session_state.llm_label = DEFAULT_LLM_LABEL

# ─────────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────────
def get_vr_url(model: str, view: str) -> str:
    return YACHT_MODELS.get(model, {}).get(view) or YACHT_MODELS["Princess Y85"]["Full"]

def detect_vr_intent(text: str) -> tuple[str, str] | None:
    t = text.lower()
    # Require explicit VR/view keywords to avoid false triggers on general questions
    if not any(k in t for k in ("vr", "virtual", "view", "show me", "tour", "display")):
        return None
    model = "Princess X95" if "x95" in t else "Princess Y85"
    view = "Compact" if "compact" in t else ("Tall" if "tall" in t else "Full")
    return model, view

def render_iframe(src: str, height: int = 600) -> None:
    if hasattr(st, "iframe"):
        st.iframe(src, height=height)
    else:
        components.iframe(src, height=height, scrolling=True)

def stream_assistant_reply(user_text: str):
    placeholder = st.empty()
    full = ""
    if groq_client is None:
        msg = f"⚠️ **API not configured:** {client_err}"
        placeholder.markdown(msg)
        return msg

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st.session_state.messages
    messages.append({"role": "user", "content": user_text})

    model_id = LLM_MODELS.get(st.session_state.llm_label, "openai/gpt-oss-20b")

    try:
        stream = groq_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.7,
            stream=True,
            max_tokens=1024,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
                placeholder.markdown(full + "▌")
        placeholder.markdown(full or "_No response from model._")
    except APIError as e:
        detail = getattr(e, "message", str(e))
        if getattr(e, "status_code", None) == 429:
            detail = f"Rate-limited by Groq free tier. Wait ~20s. ({detail})"
        full = f"⚠️ **API Error:** {detail}"
        placeholder.error(full)
    except Exception as e:
        full = f"⚠️ **Connection error:** {e}"
        placeholder.error(full)
        traceback.print_exc()
    return full

def export_chat_json() -> str:
    return json.dumps(
        {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "llm_model": LLM_MODELS.get(st.session_state.llm_label),
            "messages": st.session_state.messages,
            "vr_model": st.session_state.vr_model,
            "vr_view": st.session_state.vr_view,
        },
        indent=2,
    )

# ─────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛥️ Princess Yacht VR")
    if groq_client:
        st.success("✅ Groq connected", icon="🔌")
    else:
        st.error(f"❌ {client_err}", icon="🔌")

    st.divider()

    st.markdown("### 🧠 Language Model")
    labels = list(LLM_MODELS.keys())
    current = st.session_state.llm_label
    if current not in labels:
        current = DEFAULT_LLM_LABEL

    st.selectbox(
        "Model (free tier)",
        labels,
        index=labels.index(current),
        key="llm_label",
    )

    st.divider()

    st.markdown("### 🎛️ VR Controls")
    model_pick = st.selectbox(
        "Yacht model",
        list(YACHT_MODELS.keys()),
        index=list(YACHT_MODELS.keys()).index(st.session_state.vr_model),
    )
    view_pick = st.radio(
        "View",
        ["Compact", "Tall", "Full"],
        index=["Compact", "Tall", "Full"].index(st.session_state.vr_view),
        horizontal=True,
    )
    if st.button("▶️ Load VR", use_container_width=True, type="primary"):
        st.session_state.vr_model = model_pick
        st.session_state.vr_view = view_pick
        st.session_state.vr_url = get_vr_url(model_pick, view_pick)
        st.rerun()

    st.divider()
    if st.button("Clear chat & VR", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vr_url = None
        st.rerun()

    st.caption("Powered by Groq & Streamlit")

# ─────────────────────────────────────────────────────────────
# 7. MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
st.title("🛥️ Princess Yacht Virtual Experience")
st.caption("Ask about the **X95** or **Y85** — say *“show me the full VR view”* to enter VR.")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("💬 Chat with the Yacht Expert")
    chat_container = st.container(height=520, border=False)
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome aboard! Try a suggestion below or start typing.")
            cols = st.columns(2)
            for i, prompt in enumerate(SUGGESTED_PROMPTS):
                if cols[i % 2].button(prompt, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

        for msg in st.session_state.messages:
            avatar = "🧑‍✈️" if msg["role"] == "user" else "⚓"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    user_input = None
    if st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    typed = st.chat_input("Ask me about the X95 or Y85…")
    if typed:
        user_input = typed

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Check VR intent immediately upon receiving input
        vr_hit = detect_vr_intent(user_input)
        if vr_hit:
            m, v = vr_hit
            st.session_state.vr_model = m
            st.session_state.vr_view = v
            st.session_state.vr_url = get_vr_url(m, v)

        with chat_container:
            with st.chat_message("user", avatar="🧑‍✈️"):
                st.markdown(user_input)
            with st.chat_message("assistant", avatar="⚓"):
                reply = stream_assistant_reply(user_input)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ── RIGHT: VR VIEWER ──────────────────────────────────────────
with right:
    st.subheader("🌐 Virtual Reality Viewer")
    st.caption(f"Showing **{st.session_state.vr_model}** · **{st.session_state.vr_view}** view")
    
    if st.session_state.vr_url:
        render_iframe(st.session_state.vr_url, height=600)
        st.markdown(
            f"If the VR view is blank, [click here to open directly in a new tab]({st.session_state.vr_url}).",
            unsafe_allow_html=True
        )
    else:
        st.markdown(DEFAULT_VR_HTML, unsafe_allow_html=True)
