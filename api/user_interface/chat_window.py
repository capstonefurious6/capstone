import gradio as gr
import requests

def chat_with_bot(text, image):
    payload = {"text": text}

    files = {}
    if image:
        files['image'] = (image.name, open(image.name, 'rb'), 'image/png')

    response = requests.post(
        "http://127.0.0.1:5000/getresponse",
        data=payload,
        files=files if files else None
    )

    if response.status_code == 200:
        return response.json().get("reply", "No response.")
    else:
        return f"Error: {response.status_code}"

# Gradio UI
iface = gr.Interface(
    fn=chat_with_bot,
    inputs=[
        gr.Textbox(label="Your message"),
        gr.File(label="Upload an image", file_types=[".png", ".jpg", ".jpeg"])
    ],
    outputs="text",
    title="Medical Bot that answers the query"
)

iface.launch(share=True)