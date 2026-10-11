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

# ─────────────────────────────────────────────────────────────
# LLM MODEL OPTIONS — Verified free on OpenRouter (Oct 2026)
# All IDs below appear in OpenRouter's free catalog with $0 prompt pricing.
# Sources: OpenRouter free catalog, community-verified free lists.
# ─────────────────────────────────────────────────────────────
LLM_MODELS = {
    "🛡️ Auto (Free Router — picks a free model)": "openrouter/free",
    "Nemotron 3 Ultra 550B (reasoning, 1M)":       "nvidia/nemotron-3-ultra-550b-a55b:free",
    "Nemotron 3 Super 120B (balanced, 1M)":        "nvidia/nemotron-3-super-120b-a12b:free",
    "Qwen 3.6 Plus (1M, strong coding)":           "qwen/qwen3.6-plus:free",
    "Qwen 3 Coder 480B (1M, tool calling)":        "qwen/qwen3-coder:free",
    "Llama 3.3 70B (proven, 131K)":                "meta-llama/llama-3.3-70b-instruct:free",
    "DeepSeek R1 (reasoning, 164K)":               "deepseek/deepseek-r1:free",
    "Gemma 3 27B (vision, 131K)":                  "google/gemma-3-27b-it:free",
    "Gemma 3 12B (vision, 131K)":                  "google/gemma-3-12b-it:free",
    "Mistral Small 3.1 24B (vision, 128K)":        "mistralai/mistral-small-3.1-24b-instruct:free",
    "Step 3.5 Flash (fast, 256K)":                 "stepfun/step-3.5-flash:free",
    "GLM 4.5 Air (131K)":                          "z-ai/glm-4.5-air:free",
    "Trinity Large (131K)":                        "arcee-ai/trinity-large-preview:free",
    "Nemotron Nano 30B (256K)":                    "nvidia/nemotron-3-nano-30b-a3b:free",
    "GPT-OSS 120B (131K)":                         "openai/gpt-oss-120b:free",
    "GPT-OSS 20B (131K)":                          "openai/gpt-oss-20b:free",
    "MiniMax M2.5 (204K)":                         "minimax/minimax-m2.5:free",
}
DEFAULT_LLM_LABEL = "🛡️ Auto (Free Router — picks a free model)"

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
# 3. SECRET / CLIENT INITIALIZATION (cached)
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
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            http_client=http_client,
            default_headers={
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
    st.session_state.messages = []
if "vr_html" not in st.session_state:
    st.session_state.vr_html = DEFAULT_VR_HTML
if "vr_model" not in st.session_state:
    st.session_state.vr_model = "Princess Y85"
if "vr_view" not in st.session_state:
    st.session_state.vr_view = "Full"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "llm_label" not in st.session_state:
    st.session_state.llm_label = DEFAULT_LLM_LABEL
if "llm_models" not in st.session_state:                # ← NEW
    st.session_state.llm_models = dict(LLM_MODELS)      # ← NEW

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
    t = text.lower()
    if not any(k in t for k in ("vr", "virtual", "view", "show me", "tour")):
        if not any(k in t for k in ("x95", "y85")):
            return None
    model = "Princess X95" if "x95" in t else "Princess Y85"
    view = "Compact" if "compact" in t else ("Tall" if "tall" in t else "Full")
    return model, view


# ── NEW: Live free-model fetcher ──────────────────────────────
def fetch_free_models_live() -> dict[str, str]:
    """
    Query OpenRouter's /api/v1/models endpoint and return a dict of
    {display_label: model_id} for models whose pricing.prompt == '0'.
    Falls back to the static LLM_MODELS dict on any error.
    """
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            timeout=15.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        free = {}
        for m in payload.get("data", []):
            pricing = m.get("pricing") or {}
            # A model is free if both prompt and completion prices are 0
            if (
                str(pricing.get("prompt")) == "0"
                and str(pricing.get("completion")) == "0"
            ):
                model_id = m.get("id", "")
                if not model_id:
                    continue
                # Build a friendly label: strip ':free', add context
                name = m.get("name") or model_id
                ctx = m.get("context_length")
                label = f"{name} ({ctx:,} ctx)" if ctx else name
                free[label] = model_id
        # Always prepend the auto-router
        return {"🛡️ Auto (Free Router — picks a free model)": "openrouter/free", **free}
    except Exception as e:
        # Silently fall back — the static dict is always available
        return dict(LLM_MODELS)


# ── NEW: Refresh button handler ───────────────────────────────
def refresh_models():
    st.session_state.llm_models = fetch_free_models_live()
    # If the previously-selected label no longer exists, reset to Auto
    if st.session_state.llm_label not in st.session_state.llm_models:
        st.session_state.llm_label = DEFAULT_LLM_LABEL
    st.toast(f"Loaded {len(st.session_state.llm_models)} free models.", icon="✅")


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

    model_id = st.session_state.llm_models.get(
        st.session_state.llm_label, "openrouter/free"
    )
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
                f"Rate-limited by OpenRouter free tier (20 RPM / 50 RPD). "
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
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "llm_model": st.session_state.llm_models.get(st.session_state.llm_label),
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
        st.success("✅ OpenRouter connected", icon="🔌")
    else:
        st.error(f"❌ {client_err}", icon="🔌")

    st.divider()

    # ── Model selector (live, refreshable) ────────────────────
    st.markdown("### 🧠 Language Model")
    models_dict = st.session_state.llm_models
    labels = list(models_dict.keys())

    # Guard: selected label must exist after a refresh
    current = st.session_state.llm_label
    if current not in labels:
        current = DEFAULT_LLM_LABEL

    st.selectbox(
        "Model (free tier)",
        labels,
        index=labels.index(current),
        key="llm_label",
        help="All models listed have $0 prompt pricing on OpenRouter. "
             "Use 'Auto' to let OpenRouter pick a working free model.",
    )

    # ── Refresh button ─────────────────────────────────────────
    if st.button(
        "🔄 Refresh free models",
        use_container_width=True,
        help="Queries OpenRouter /api/v1/models and filters for pricing.prompt == '0'.",
    ):
        refresh_models()
        st.rerun()

    st.caption(f"Loaded **{len(models_dict)}** free models.")

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
                st.session_state.vr_html = generate_vr_html(m, v)

            with st.chat_message("assistant", avatar="⚓"):
                reply = stream_assistant_reply(user_input)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

with right:
    st.subheader("🌐 Virtual Reality Viewer")
    st.caption(
        f"Showing **{st.session_state.vr_model}** · **{st.session_state.vr_view}** view"
    )
    components.html(st.session_state.vr_html, height=680, scrolling=True)
