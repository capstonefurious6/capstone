import streamlit as st
from main import process_query, chat_with_gemini, chat_history
import os


def save_uploaded_file(uploaded_file, text):
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, "uploaded_image.png")
    text_path = os.path.join(save_dir, "uploaded_text.txt")

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with open(text_path, "w") as f:
        f.write(text)

    return image_path, text_path


st.title("Medical Insights On Uploaded Image and Report")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
text_input = st.text_area("Enter some text")

if uploaded_file and text_input:
    if st.button("Get Medical Insights"):
        image_path, text_path = save_uploaded_file(uploaded_file, text_input)
        st.success("File uploaded successfully!")

        with st.spinner("Processing..."):
            medical_context, medical_response, general_response, chat_history_ref, api_key = process_query(image_path, text_path)
            st.session_state.chat_started = True
            st.session_state.api_key = api_key
            st.session_state.context = medical_context

        st.success("Analysis Complete!")

        st.title("BioGPT diagnosis")
        st.markdown(medical_context)
        st.title("Medical detailed report")
        st.markdown(medical_response)
        st.title("General Report")
        st.markdown(general_response)


# Chat interface
if st.session_state.get("chat_started", False):
    st.title("Medical Chat Assistant")

    for i, msg in enumerate(chat_history):
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["assistant"])

    user_input = st.chat_input("Ask a follow-up question or clarification...")

    if user_input:
        with st.spinner("Thinking..."):
            response = chat_with_gemini(st.session_state.api_key, user_input)
            st.rerun()

