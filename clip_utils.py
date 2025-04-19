import torch
import open_clip
from torchvision import transforms
from PIL import Image

torch.classes.__path__ = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, transform = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def get_clip_embeddings(image_paths, texts):
    images = torch.stack([load_image(img) for img in image_paths]).to(device)
    tokenized_texts = tokenizer(texts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokenized_texts)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return image_features, text_features
