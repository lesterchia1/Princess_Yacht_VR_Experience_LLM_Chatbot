import gradio as gr
import os
from openai import OpenAI, APIError
import httpx
import traceback

# --- API Initialization ---
hf_token = os.getenv("HF_TOKEN")
print(f"HF_TOKEN present: {hf_token is not None}")
if hf_token:
    print(f"HF_TOKEN length: {len(hf_token)}")

if not hf_token:
    print("CRITICAL: HF_TOKEN is missing.")
    hf_client = None
else:
    try:
        custom_http_client = httpx.Client(timeout=60.0) 
        hf_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
            http_client=custom_http_client
        )
        print("HF client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize HF client: {e}")
        hf_client = None

# --- VR CONFIGURATION ---
yacht_models = {
    "Princess Y85": {
        "Compact": "https://vrcloud.com/?pv=1550754573&play=1",
        "Tall":    "https://vrcloud.com/?pv=1550754573&play=1",
        "Full":    "https://vrcloud.com/?pv=1550754573&play=1"
    },
    "Princess X95": {
        "Compact": "https://vrcloud.com/?pv=1620886782&play=1",
        "Tall":    "https://vrcloud.com/?pv=1620886782&play=1",
        "Full":    "https://vrcloud.com/?pv=1620886782&play=1"
    }
}

DEFAULT_VR_HTML = """
<div style="text-align:center; padding: 20px; border: 1px dashed #ccc; border-radius: 5px;">
    <h3>🚢 Ask the chat to start the Virtual Reality Experience!</h3>
    <p>Try: <b>Show me the X95 full VR view</b></p>
</div>
"""

def generate_vr_html(model, view):
    try:
        link = yacht_models[model][view]
    except (KeyError, TypeError):
        link = yacht_models["Princess Y85"]["Full"]
    
    return f"""
    <div style="text-align:center">
        <h3 style="color: #2c3e50;">🚢 {model} – {view} VR View</h3>
        <iframe src="{link}" width="100%" height="600px" style="border:none; border-radius:10px;"></iframe>
    </div>
    """

# --- CHAT LOGIC ---
def yacht_chatbot(user_input, chat_history):
    if chat_history is None:
        chat_history = []

    # VR switching logic (works regardless of API status)
    text = user_input.lower()
    model_choice = "Princess X95" if "x95" in text else "Princess Y85"
    view_choice = "Compact" if "compact" in text else ("Tall" if "tall" in text else "Full")
    vr_html = generate_vr_html(model_choice, view_choice)

    # Check API client
    if hf_client is None:
        chat_history.append(gr.ChatMessage(role="user", content=user_input))
        chat_history.append(gr.ChatMessage(role="assistant", content="⚠️ Error: Hugging Face API token (HF_TOKEN) is not configured. Please set your HF_TOKEN environment variable."))
        yield chat_history, vr_html
        return

    # Build messages for OpenAI API
    messages = [{"role": "system", "content": "You are a helpful yacht expert specializing in Princess Yachts."}]
    for msg in chat_history:
        if isinstance(msg, gr.ChatMessage):
            messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg:
            messages.append(msg)
    messages.append({"role": "user", "content": user_input})

    # Add user message to chat history
    chat_history.append(gr.ChatMessage(role="user", content=user_input))
    chat_history.append(gr.ChatMessage(role="assistant", content=""))
    
    try:
        print(f"Calling API with model: meta-llama/Llama-3.1-8B-Instruct")
        stream = hf_client.chat.completions.create(
            #model="Qwen/Qwen2.5-7B-Instruct", #meta-llama/Llama-3.1-8B-Instruct
            model="meta-llama/Llama-3.1-8B-Instruct",  # Updated serverless model string
            messages=messages,
            temperature=0.7,
            stream=True,
            max_tokens=1024
        )
        
        token_accumulator = ""
        chunk_count = 0
        
        
        #for chunk in stream:
        #    chunk_count += 1
            # Check that choices is not empty and delta has content
        #    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        #        token_accumulator += chunk.choices[0].delta.content
        #        chat_history[-1].content = token_accumulator
        #        yield chat_history, vr_html
        #print(f"Stream completed. Received {chunk_count} chunks.")
        
        
        
        
        for chunk in stream:
            chunk_count += 1
            if chunk.choices[0].delta.content:
                token_accumulator += chunk.choices[0].delta.content
                chat_history[-1].content = token_accumulator
                yield chat_history, vr_html
        
        print(f"Stream completed. Received {chunk_count} chunks.")
                
    except APIError as e:
        error_msg = f"API Error: {e.message if hasattr(e, 'message') else str(e)}"
        print(f"API Error details: {error_msg}")
        chat_history[-1].content = error_msg
        yield chat_history, vr_html
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        print(f"Exception: {error_msg}")
        print(traceback.format_exc())
        chat_history[-1].content = error_msg
        yield chat_history, vr_html

def clear_all():
    return [], DEFAULT_VR_HTML, ""

# --- UI DEFINITION ---
with gr.Blocks(title="Princess Yacht VR") as demo:
    gr.Markdown("# 🛥️ Princess Yacht Virtual Experience")
    
    dynamic_vr_html = gr.HTML(DEFAULT_VR_HTML)
    chatbot = gr.Chatbot(label="Princess Yacht Chat", height=450)
    msg = gr.Textbox(placeholder="Ask me about the X95 or Y85...", label="Your Message")
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear Chat/VR", scale=1)

    # Bind Events
    submit_btn.click(yacht_chatbot, [msg, chatbot], [chatbot, dynamic_vr_html]).then(lambda: "", None, [msg])
    msg.submit(yacht_chatbot, [msg, chatbot], [chatbot, dynamic_vr_html]).then(lambda: "", None, [msg])
    clear.click(clear_all, None, [chatbot, dynamic_vr_html, msg])

if __name__ == "__main__":
    demo.launch()
