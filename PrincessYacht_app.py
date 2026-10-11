"""
Princess Yacht VR Experience — Streamlit edition
Migrated from Gradio. Now powered by Groq. Requires Python 3.10+.
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
# 1. PAGE CONFIG (must be first Streamlit call)
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

# ─────────────────────────────────────────────────────────────
# LLM MODEL OPTIONS — Groq (verified free tier, Oct 2026)
# llama-3.1-8b-instant was shut down for free/developer tiers on 2026-08-16.
# Groq's official recommended replacement is openai/gpt-oss-20b.
# ─────────────────────────────────────────────────────────────
LLM_MODELS = {
    "🧠 GPT-OSS 20B (fast reasoning, 1,000 RPD)":     "openai/gpt-oss-20b",
    "🧠 GPT-OSS 120B (stronger reasoning, 1,000 RPD)": "openai/gpt-oss-120b",
    "🚀 Groq Compound (multi-model system, 250 RPD)":  "groq/compound",
    "⚡ Qwen 3.6 27B (high TPM, 1,000 RPD)":           "qwen/qwen3.6-27b",
    "👁️ Qwen 3.8 27B (vision, 1,000 RPD)":             "qwen/qwen3.8-27b",
    "🛡️ GPT-OSS Safeguard 20B (safety-focused)":       "openai/gpt-oss-safeguard-20b",
}
DEFAULT_LLM_LABEL = "🧠 GPT-OSS 20B (fast reasoning, 1,000 RPD)"

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
    "enthusiastic, and factual. If the user asks to see a VR view, "
    "acknowledge it briefly — the UI will handle the switch."
)

SUGGESTED_PROMPTS = [
    "Tell me about the Princess X95",
    "Show me the X95 full VR view",
    "Compare X95 vs Y85",
    "What's the top speed of the Y85?",
]

# ─────────────────────────────────────────────────────────────
# 3. SECRET / CLIENT INITIALIZATION (cached)   ← CHANGED: Groq
# ─────────────────────────────────────────────────────────────
def _get_groq_key() -> str | None:
    """Read Groq API key from Streamlit secrets, fall back to env var."""
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
        return None, "GROQ_API_KEY is missing (checked st.secrets and env)."
    try:
        http_client = httpx.Client(timeout=60.0)
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",       # ← Groq base URL
            api_key=key,
            http_client=http_client,
        )
        return client, None
    except Exception as e:
        return None, f"Client init failed: {e}"


or_client, client_err = get_groq_client()      # variable name kept for minimal diff

# ─────────────────────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vr_html" not in st.session_state:
    st.session_state.vr_html = DEFAULT_VR_HTML
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
    """Return the VR URL for the given model/view, with fallback."""
    return (
        YACHT_MODELS.get(model, {}).get(view)
        or YACHT_MODELS["Princess Y85"]["Full"]
    )


def detect_vr_intent(text: str) -> tuple[str, str] | None:
    t = text.lower()
    if not any(k in t for k in ("vr", "virtual", "view", "show me", "tour")):
        if not any(k in t for k in ("x95", "y85")):
            return None
    model = "Princess X95" if "x95" in t else "Princess Y85"
    view = "Compact" if "compact" in t else ("Tall" if "tall" in t else "Full")
    return model, view


def render_iframe(src: str, height: int = 680) -> None:
    """Render a URL as an iframe — uses st.iframe if available, else fallback."""
    if hasattr(st, "iframe"):
        st.iframe(src, height=height)
    else:
        components.iframe(src, height=height, scrolling=True)


def stream_assistant_reply(user_text: str) -> str:
    placeholder = st.empty()
    full = ""
    if or_client is None:
        msg = f"⚠️ **API not configured:** {client_err}"
        placeholder.markdown(msg)
        return msg

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st.session_state.messages
    messages.append({"role": "user", "content": user_text})

    # ← CHANGED: direct lookup from static dict
    model_id = LLM_MODELS.get(st.session_state.llm_label, "openai/gpt-oss-20b")

    try:
        stream = or_client.chat.completions.create(
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
            detail = (
                f"Rate-limited by Groq free tier (30 RPM / 1,000 RPD). "
                f"Wait ~20s or switch model. ({detail})"
            )
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
            "llm_model": LLM_MODELS.get(st.session_state.llm_label),  # ← direct lookup
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
    if or_client:
        st.success("✅ Groq connected", icon="🔌")     # ← CHANGED
    else:
        st.error(f"❌ {client_err}", icon="🔌")

    st.divider()

    # ── Model selector (static — Groq models are stable) ──────
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
        help="All models below are available on Groq's permanent free tier. "
             "No credit card required.",
    )
    st.caption("⚡ Groq free tier — no live refresh needed.")

    st.divider()

    # ── VR controls ────────────────────────────────────────────
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
        st.session_state.vr_html = DEFAULT_VR_HTML
        st.rerun()

    st.divider()
    st.markdown("### 🧹 Session")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vr_html = DEFAULT_VR_HTML
            st.session_state.vr_url = None
            st.rerun()
    with col_b:
        st.download_button(
            "Export JSON",
            data=export_chat_json(),
            file_name=f"yacht_chat_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption("Migrated from Gradio → Streamlit · Powered by Groq")

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
        with chat_container:
            with st.chat_message("user", avatar="🧑‍✈️"):
                st.markdown(user_input)

            vr_hit = detect_vr_intent(user_input)
            if vr_hit:
                m, v = vr_hit
                st.session_state.vr_model = m
                st.session_state.vr_view = v
                st.session_state.vr_url = get_vr_url(m, v)

            with st.chat_message("assistant", avatar="⚓"):
                reply = stream_assistant_reply(user_input)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ── RIGHT: VR ─────────────────────────────────────────────────
with right:
    st.subheader("🌐 Virtual Reality Viewer")
    st.caption(
        f"Showing **{st.session_state.vr_model}** · **{st.session_state.vr_view}** view"
    )
    if st.session_state.vr_url:
        render_iframe(st.session_state.vr_url, height=600)
        st.caption(f"[Open in new tab ↗]({st.session_state.vr_url})")
    else:
        st.markdown(st.session_state.vr_html, unsafe_allow_html=True)
