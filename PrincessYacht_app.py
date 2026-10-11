"""
Princess Yacht VR Experience — Streamlit edition
Migrated from Gradio. Now powered by OpenRouter. Requires Python 3.10+.
"""

import os
import json
import traceback
from datetime import datetime

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

# OpenRouter free-tier models (https://openrouter.ai/models?max_price=0)
# ← NEW: model selector map. All suffixed with ":free" → no billing.
LLM_MODELS = {
    "Llama 3.1 8B (fast, recommended)": "meta-llama/llama-3.1-8b-instruct:free",
    "Llama 3.3 70B (smarter, slower)":  "meta-llama/llama-3.3-70b-instruct:free",
    "Qwen 2.5 7B":                      "qwen/qwen-2.5-7b-instruct:free",
    "Gemma 3 27B":                      "google/gemma-3-27b-it:free",
    "DeepSeek Chat v3":                 "deepseek/deepseek-chat-v3-0324:free",
}
DEFAULT_LLM_LABEL = "Llama 3.1 8B (fast, recommended)"

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
# 3. SECRET / CLIENT INITIALIZATION (cached)   ← CHANGED
# ─────────────────────────────────────────────────────────────
def _get_openrouter_key() -> str | None:
    """Read OpenRouter key from Streamlit secrets, fall back to env var."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY")


@st.cache_resource(show_spinner=False)
def get_openrouter_client():
    key = _get_openrouter_key()
    if not key:
        return None, "OPENROUTER_API_KEY is missing (checked st.secrets and env)."

    try:
        http_client = httpx.Client(timeout=60.0)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",       # ← CHANGED
            api_key=key,
            http_client=http_client,
            default_headers={
                # Optional but recommended by OpenRouter — shows up in
                # your dashboard analytics. Harmless if left in a public repo.
                "HTTP-Referer": "https://share.streamlit.io",
                "X-Title": "Princess Yacht VR",
            },
        )
        return client, None
    except Exception as e:
        return None, f"Client init failed: {e}"


or_client, client_err = get_openrouter_client()

# ─────────────────────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list[{"role","content"}]
if "vr_html" not in st.session_state:
    st.session_state.vr_html = DEFAULT_VR_HTML
if "vr_model" not in st.session_state:
    st.session_state.vr_model = "Princess Y85"
if "vr_view" not in st.session_state:
    st.session_state.vr_view = "Full"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "llm_label" not in st.session_state:                 # ← NEW
    st.session_state.llm_label = DEFAULT_LLM_LABEL

# ─────────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────────
def generate_vr_html(model: str, view: str) -> str:
    link = YACHT_MODELS.get(model, {}).get(view) \
           or YACHT_MODELS["Princess Y85"]["Full"]
    return f"""
    <div style="text-align:center">
        <h3 style="color:#2c3e50; margin-bottom:6px;">🚢 {model} – {view} VR View</h3>
        <p style="font-size:12px; color:#607d8b; margin-top:0;">
            <a href="{link}" target="_blank" rel="noopener">Open in new tab ↗</a>
        </p>
        <iframe src="{link}" width="100%" height="600px"
                style="border:none; border-radius:10px;"
                allow="xr-spatial-tracking; fullscreen; gyroscope; accelerometer"
                allowfullscreen></iframe>
    </div>
    """


def detect_vr_intent(text: str) -> tuple[str, str] | None:
    """Return (model, view) if the text mentions a VR switch, else None."""
    t = text.lower()
    if not any(k in t for k in ("vr", "virtual", "view", "show me", "tour")):
        # still allow pure model mentions to switch
        if not any(k in t for k in ("x95", "y85")):
            return None
    model = "Princess X95" if "x95" in t else "Princess Y85"
    view = "Compact" if "compact" in t else ("Tall" if "tall" in t else "Full")
    return model, view


def stream_assistant_reply(user_text: str) -> str:
    """Stream tokens into the current st.chat_message context. Returns full text."""
    placeholder = st.empty()
    full = ""
    if or_client is None:
        msg = f"⚠️ **API not configured:** {client_err}"
        placeholder.markdown(msg)
        return msg

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st.session_state.messages          # already contains history
    messages.append({"role": "user", "content": user_text})

    model_id = LLM_MODELS[st.session_state.llm_label]    # ← CHANGED
    try:
        stream = or_client.chat.completions.create(      # ← CHANGED
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
        # OpenRouter puts useful info in e.body / e.message
        detail = getattr(e, "message", str(e))
        if getattr(e, "status_code", None) == 429:
            detail = (
                f"Rate-limited by OpenRouter free tier. "
                f"Wait ~20s or switch model in the sidebar. ({detail})"
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
            "exported_at": datetime.utcnow().isoformat() + "Z",
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
    if or_client:
        st.success("✅ OpenRouter connected", icon="🔌")   # ← CHANGED
    else:
        st.error(f"❌ {client_err}", icon="🔌")

    st.divider()

    # ← NEW: model selector
    st.markdown("### 🧠 Language Model")
    st.selectbox(
        "Model (free tier)",
        list(LLM_MODELS.keys()),
        index=list(LLM_MODELS.keys()).index(st.session_state.llm_label),
        key="llm_label",
        help="All models below are OpenRouter free-tier (no billing).",
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
        st.session_state.vr_html = generate_vr_html(model_pick, view_pick)
        st.rerun()

    st.divider()
    st.markdown("### 🧹 Session")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vr_html = DEFAULT_VR_HTML
            st.rerun()
    with col_b:
        st.download_button(
            "Export JSON",
            data=export_chat_json(),
            file_name=f"yacht_chat_{datetime.utcnow():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption("Migrated from Gradio → Streamlit · Powered by OpenRouter")

# ─────────────────────────────────────────────────────────────
# 7. MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
st.title("🛥️ Princess Yacht Virtual Experience")
st.caption("Ask about the **X95** or **Y85** — say *“show me the full VR view”* to enter VR.")

left, right = st.columns([3, 2], gap="large")

# ── LEFT: Chat ────────────────────────────────────────────────
with left:
    st.subheader("💬 Chat with the Yacht Expert")

    chat_container = st.container(height=520, border=False)
    with chat_container:
        if not st.session_state.messages:
            st.info(
                "👋 Welcome aboard! Try one of the suggestions below or just start typing."
            )
            cols = st.columns(2)
            for i, prompt in enumerate(SUGGESTED_PROMPTS):
                if cols[i % 2].button(prompt, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

        for msg in st.session_state.messages:
            avatar = "🧑‍✈️" if msg["role"] == "user" else "⚓"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # Handle pending suggestion (from chip click)
    user_input = None
    if st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    typed = st.chat_input("Ask me about the X95 or Y85…")
    if typed:
        user_input = typed

    # Process user turn
    if user_input:
        # 1) Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user", avatar="🧑‍✈️"):
                st.markdown(user_input)

            # 2) VR intent detection — updates right panel
            vr_hit = detect_vr_intent(user_input)
            if vr_hit:
                m, v = vr_hit
                st.session_state.vr_model = m
                st.session_state.vr_view = v
                st.session_state.vr_html = generate_vr_html(m, v)

            # 3) Stream assistant reply
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
    components.html(st.session_state.vr_html, height=680, scrolling=True)
