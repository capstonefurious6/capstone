from clip_utility import CLIP_Prompt 
import torch 
import clip
from PIL import Image

clip_model = CLIP_Prompt() 
image_path = "api/chestxrayPnemonia.png"
img = Image.open(image_path)
print(img)
prompt_from_image = clip_model.get_prompt(img)
print(prompt_from_image)