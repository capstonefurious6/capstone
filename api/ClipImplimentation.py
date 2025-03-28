import torch 
import clip
from PIL import Image
print("import is completed")
device = 'mps' if torch.backends.mps.is_available else 'cpu'
print(device)
clip_model,preprocess = clip.load('ViT-B/32',device=device) 

def get_prompt(img,text_prompts,clip_model,device):
    text_tokens = clip.tokenize(text_prompts).to(device)
    img = preprocess(img).unsqueeze(0).to(device)
    print("we are just about to call the with code word")
    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
    best_match = text_prompts[similarity.argmax()]
    return best_match 

#reading the image from the folder - later this would be dynamically created. 
image_path = "pexels-pixabay-45201.jpg"
img = Image.open(image_path) 

#setting up the text prompt - here 
text_prompts = ['lung scan is all good','lung scan with pnuemonia, do let me know about the treatment','kidney scan which looks okay, talk about health benifits','brain scan','It is about a cat']

#calling the method like it is no ones business. 
print(get_prompt(img,text_prompts,clip_model,device))


    