# Princess_Yacht_VR_Experience_LLM_Chatbot
Chat about Princess Yachts and view VR tours 


# 🛥️ Princess Yacht VR — Streamlit
Chat with a Princess Yachts expert and launch an immersive VR tour — all in one page.

## Features
- 💬 Streaming LLM chat (Llama 3.1 8B via HF Router)
- 🌐 Inline VR iframe triggered by chat *or* sidebar
- 📥 One-click JSON chat export
- 🔐 Secrets via Streamlit Cloud (`HF_TOKEN`)

## Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
echo 'HF_TOKEN = "hf_xxx"' > .streamlit/secrets.toml
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push repo to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick repo + `app.py`.
3. In **Settings → Secrets**, paste:
   ```toml
   HF_TOKEN = "hf_xxx"
   ```
4. Deploy.

## Notes
- If `vrcloud.com` blocks iframe embedding (`X-Frame-Options`), use the
  **"Open in new tab ↗"** link rendered above the iframe.
- The OpenAI client is `@st.cache_resource`-cached → no re-initialization on reruns.
