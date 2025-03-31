import pandas as pd
import numpy as np
import os
import faiss
import torch
import open_clip  # CLIP from OpenAI
from torchvision import transforms
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    return transform(image)

def get_clip_embeddings(image_paths, texts, model, tokenizer, device):
    images = torch.stack([load_image(img) for img in image_paths]).to(device)
    tokenized_texts = tokenizer(texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokenized_texts)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for cosine similarity
    index.add(embeddings)
    return index

def hybrid_search(query_image_path, query_text, image_index, text_index, texts, image_weight=0.6, text_weight=0.4, top_k=2):
    query_image_embedding, query_text_embedding = get_clip_embeddings([query_image_path], [query_text], model, tokenizer, device)
    
    _, img_indices = image_index.search(query_image_embedding, top_k)
    _, text_indices = text_index.search(query_text_embedding, top_k)
    
    combined_scores = {}
    
    for i, img_idx in enumerate(img_indices[0]):
        combined_scores[img_idx] = combined_scores.get(img_idx, 0) + image_weight * (1 - i / top_k)
    
    for i, txt_idx in enumerate(text_indices[0]):
        combined_scores[txt_idx] = combined_scores.get(txt_idx, 0) + text_weight * (1 - i / top_k)
    
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [(texts[idx], score) for idx, score in sorted_results[:top_k]]


def query_gemini(api_key, query_text):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(query_text)
    return response.text

dataset_path = "D:/Mini Project/data/NLP_aug_datasets/df_train_aug.csv"
df = pd.read_csv(dataset_path)
path = "D:/Mini Project/data/mimic_dset/re_512_3ch/Train/"
df["image_name"] = df["id"].astype(str) + ".jpg"
df["image_path"] = df["image_name"].apply(lambda x: os.path.join(path, x))
df1 = df[['id', 'text', 'image_path']][df["image_path"].apply(os.path.exists)].reset_index(drop=True)

global device, transform
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, transform = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

batch_size = 16
sample_images = df1["image_path"][:batch_size].tolist()
sample_texts = df1["text"][:batch_size].tolist()

image_embeds, text_embeds = get_clip_embeddings(sample_images, sample_texts, model, tokenizer, device)
image_index = build_faiss_index(image_embeds)
text_index = build_faiss_index(text_embeds)

print("FAISS Index Built!")

query_image_path = df1["image_path"][165]
query_text = df1["text"][165]

print("Image path = ",query_image_path)
print("Report = ",query_text)

top_matches = hybrid_search(query_image_path, query_text, image_index, text_index,sample_texts)
best_matching_text = top_matches[0][0] + top_matches[1][0]

prompt = "Give Medical Report and also Give medical advice on how to fix the condition. " \
        "Give a rating on 1-5 on how serious the given condition is- "+ best_matching_text

api_key = "AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ"
gemini_response = query_gemini(api_key, prompt)

print("🔹 Gemini Response:")
print(gemini_response)


    