import os
import re
import pandas as pd
from clip_utils import get_clip_embeddings, model, tokenizer, device
from faiss_utils import build_faiss_index, hybrid_search
from bio_gpt import query_biogpt
from gemini_api import query_gemini

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_dataset():
    dataset_path = "/Users/apple/Documents/karthikl/capstone/Data_Set/df_train.csv"
    df = pd.read_csv(dataset_path)
    path = "/Users/apple/Documents/karthikl/capstone/Data_Set/re_512_3ch/Train/"
    df["image_name"] = df["id"].astype(str) + ".jpg"
    df["image_path"] = df["image_name"].apply(lambda x: os.path.join(path, x))
    return df[['id', 'text', 'image_path']][df["image_path"].apply(os.path.exists)].reset_index(drop=True)


df1 = load_dataset()

sample_images = df1["image_path"][:16].tolist()
sample_texts = df1["text"][:16].tolist()

image_embeds, text_embeds = get_clip_embeddings(sample_images, sample_texts)
image_index = build_faiss_index(image_embeds)
text_index = build_faiss_index(text_embeds)

# Initialize chat history as a list of dicts (if needed globally)
chat_history = []


def update_chat_history(user_message, assistant_message):
    chat_history.append({"user": user_message, "assistant": assistant_message})


def chat_with_gemini(api_key, user_message):
    # Use previous medical context + full chat history as context
    context = ""
    for msg in chat_history:
        context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"

    context += f"User: {user_message}\nAssistant:"

    response = query_gemini(api_key=api_key, query_text=context)
    update_chat_history(user_message, response)
    return response


def process_query(image_path, text_path):
    with open(text_path, "r") as f:
        query_text = f.read()

    top_matches = hybrid_search(image_path, query_text, image_index, text_index, sample_texts)
    best_matching_text = top_matches[0][0] + top_matches[1][0]

    observation = best_matching_text

    # prompt = f"Give Medical Report and medical advice on fixing the condition. Rate severity (1-5): {best_matching_text}"

    #Gemini API KEY
    api_key = "AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ"

    # Prompt to BioGPT
    bio_gpt_prompt = (
        f"Patient findings: {best_matching_text}\n\n"
        f"### Medical Report\n"
        f"Diagnosis:"
    )

    biogpt_diagnosis = query_biogpt(bio_gpt_prompt)
    biogpt_diagnosis = re.sub(r"(A|The) \d{1,3}-year-old (man|woman|male|female).*?[\.\n]", "", biogpt_diagnosis)
    medical_context = f'''
    Observation: {best_matching_text} \n\n
    Diagnosis: {biogpt_diagnosis}
'''

    med_prompt = (
        f"Patient findings: {medical_context}\n"
        f"Generate a medical report including diagnosis, severity (1-5), possible causes, "
        f"treatment recommendations, and next steps."
    )

    general_prompt = f"Explain in layman's terms what the following findings mean and how serious this is: {medical_context}"

    medical_response = query_gemini(api_key=api_key, query_text=med_prompt)
    general_response = query_gemini(api_key=api_key, query_text=general_prompt)

    # Add initial medical context to chat history
    update_chat_history("Initial medical report request", medical_response)

    return medical_context, medical_response, general_response, chat_history, api_key

if __name__ == "__main__":
    query_image = "uploads/uploaded_image.png"
    query_text_file = "uploads/uploaded_text.txt"
    print(process_query(query_image, query_text_file))