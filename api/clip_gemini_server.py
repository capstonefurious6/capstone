import google.generativeai as genai
import torch 
import clip
from PIL import Image
print("import is completed")
device = torch.device('cpu')
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
image_path = "api/chestxrayPnemonia.png"
img = Image.open(image_path) 

#setting up the text prompt - here 
text_prompts = ['lung scan is all good','lung scan with pnuemonia, do let me know about the treatment','kidney scan which looks okay, talk about health benifits','brain scan','It is about a cat']

#calling the method like it is no ones business. 
prompt = get_prompt(img,text_prompts,clip_model,device)
#gemini-1.5-flash 
#AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ
#setting up the api-key to interact with gemini-flash that is hosted on google systems. 
#we will access through the API simply because of the computation that is required for this.
genai.configure(api_key='AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ')
#configuring the model 
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
response = model.generate_content(prompt)
print(str(response.parts[0].text)) 