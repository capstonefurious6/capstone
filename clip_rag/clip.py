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

def search_faiss(query_vector, index, image_id_to_text, top_k=3):
    query_vector = query_vector.cpu().numpy().reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [(image_id_to_text[indices[0][i]], 1 - distances[0][i]) for i in range(top_k)]

def build_faiss_index(image_embeds, sample_texts):
    index = faiss.IndexFlatL2(image_embeds.shape[1])
    index.add(image_embeds.cpu().numpy())
    return index, {i: text for i, text in enumerate(sample_texts)}

def get_best_matching_text(sample_images, sample_texts, image_embeds, text_embeds):
    cosine_sim = (image_embeds @ text_embeds.T).cpu().numpy()
    best_matches = np.argmax(cosine_sim, axis=1)
    return [(sample_texts[best_matches[i]], cosine_sim[i][best_matches[i]]) for i in range(len(sample_images))]

def query_gemini(api_key, query_text):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(query_text)
    return response.text

def main():
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
    index, image_id_to_text = build_faiss_index(image_embeds, sample_texts)
    
    print("FAISS Index Built!")
    
    query_image_path = df1["image_path"][5]
    qery_text = df1["text"][5]
    query_embedding, _ = get_clip_embeddings([query_image_path], [qery_text], model, tokenizer, device)
    
    top_matches = search_faiss(query_embedding[0], index, image_id_to_text)
    best_matching_text = top_matches[0][0] + top_matches[1][0]
    
    api_key = "AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ"
    gemini_response = query_gemini(api_key, best_matching_text)
    
    print("🔹 Gemini Response:")
    print(gemini_response)

if __name__ == "__main__":
    main()