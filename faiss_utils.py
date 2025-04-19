import faiss
from clip_utils import get_clip_embeddings, model, tokenizer, device

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def hybrid_search(query_image_path, query_text, image_index, text_index, texts, image_weight=0.6, text_weight=0.4, top_k=2):
    query_image_embedding, query_text_embedding = get_clip_embeddings([query_image_path], [query_text])
    
    _, img_indices = image_index.search(query_image_embedding, top_k)
    _, text_indices = text_index.search(query_text_embedding, top_k)
    
    combined_scores = {}
    for i, img_idx in enumerate(img_indices[0]):
        combined_scores[img_idx] = combined_scores.get(img_idx, 0) + image_weight * (1 - i / top_k)
    
    for i, txt_idx in enumerate(text_indices[0]):
        combined_scores[txt_idx] = combined_scores.get(txt_idx, 0) + text_weight * (1 - i / top_k)
    
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [(texts[idx], score) for idx, score in sorted_results[:top_k]]
