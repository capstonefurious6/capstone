import torch 
import clip
from PIL import Image
device = 'mps' if torch.backends.mps.is_available else 'cpu'

class CLIP_Prompt: 
    clip_model = None
    preprocess = None
    text_promts = None
    text_tokens = None
    
    def __init__(self,name='default_name'): 
        print("device_initialized to", device) 
        clip_model,preprocess = clip.load('ViT-B/32',device=device) 
        self.clip_model = clip_model 
        self.preprocess = preprocess 
        self.text_prompts = [
            'lung scan is all good',
            'lung scan with pnuemonia, do let me know about the treatment',
            'kidney scan which looks okay, talk about health benifits',
            'brain scan','It is about a cat'
            ]
        self.text_tokens = clip.tokenize(self.text_prompts).to(device)
        print("initializations are done")
        
    
    def get_prompt(self,img):
        img = self.preprocess(img).unsqueeze(0).to(device)
        print("we are just about to call the with code word")
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img)
            text_features = self.clip_model.encode_text(self.text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        if similarity.argmax() > 0.5: 
            best_match = self.text_prompts[similarity.argmax()]
        else:
            best_match = ""
        return best_match 