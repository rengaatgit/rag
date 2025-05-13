import gradio as gr
import time

def upload_files(files):
    file_names = [file.name for file in files]
    return f"📁 Uploaded {len(file_names)} files: {', '.join(file_names)}"

def add_url(history, url):
    if url.strip() != "":
        return history + [[None, f"🔗 URL added: {url}"]]
    return history

def add_message(history, message):
    if message.strip() != "":
        return history + [[message, None]]
    return history

def bot_response(history):
    response = "This is a sample response"  # Replace with your actual response logic
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history

def handle_upload(history, files):
    if files:
        status = upload_files(files)
        history = history + [[None, status]]
    return history

with gr.Blocks() as demo:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        # Left side buttons (vertical stack)
        with gr.Column(scale=1, min_width=60):
            with gr.Group():
                upload_btn = gr.UploadButton(
                    "➕ File",
                    file_types=["file", "text", "image"],
                    file_count="multiple",
                    size="sm"
                )
                url_btn = gr.Button("🌐 URL", size="sm")
                url_input = gr.Textbox(
                    placeholder="Paste URL...",
                    visible=False,
                    label="URL Input",
                    scale=0
                )
        
        # Middle chat input
        with gr.Column(scale=8):
            msg = gr.Textbox(show_label=False, placeholder="Type your message...")
        
        # Right send button
        with gr.Column(scale=1, min_width=80):
            send_btn = gr.Button("Send", variant="primary")

    # Toggle URL input visibility
    def toggle_url_input():
        return {"visible": True, "__type__": "update"}
    
    url_btn.click(
        lambda: gr.update(visible=True),
        outputs=url_input
    )

    # URL submission handling
    url_input.submit(
        add_url,
        [chatbot, url_input],
        chatbot
    ).then(
        lambda: gr.update(value="", visible=False),
        outputs=url_input
    )

    # File upload handling
    upload_btn.upload(
        handle_upload,
        [chatbot, upload_btn],
        chatbot,
        queue=False
    )

    # Message handling
    send_btn.click(
        add_message,
        [chatbot, msg],
        chatbot,
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )
    
    msg.submit(
        add_message,
        [chatbot, msg],
        chatbot,
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )

if __name__ == "__main__":
    demo.launch()
