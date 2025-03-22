import streamlit as st
from PIL import Image
import io

def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_path});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_model():
    # Placeholder for loading your model
    # Replace with actual model loading code
    return None

def process_image(image, model):
    # Placeholder for image processing
    # Replace with actual image processing code
    return image

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid username or password")

def main():
    set_background(r'C:\Users\Karthikl\OneDrive - Advanced Micro Devices Inc\Documents\Personal\PES_U\Sem3\stream_litCode\Doc1.jpeg')  # Replace with your image path

    if 'logged_in' not in st.session_state:
        st.header("Integration of AI for Multi-Modal Data Fusion in Healthcare")
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        st.header("Integration of AI for Multi-Modal Data Fusion in Healthcare")
        st.title("Upload the scanned image for the AI generated report")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            model = load_model()
            processed_image = process_image(image, model)
            
            st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
