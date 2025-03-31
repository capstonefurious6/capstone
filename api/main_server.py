from flask import Flask, request, jsonify 
from clip_utility import CLIP_Prompt 
from gemini_utility import Flash
import torch 
import clip
from PIL import Image
app = Flask(__name__)

#incase of back-up to read the image directly from the folder instead. 
    # image_path = "chestxrayPnemonia.png"
    # img = Image.open(image_path)

#initializing the custom clip module. 
clip_model = CLIP_Prompt()
device = torch.device("cpu")


@app.route('/getresponse',methods=['POST'])
def responseFromGeminiFlash():
    prompt_list = {} 
    if 'image' not in request.files:
        prompt_from_image = ""
    image_file = request.files['image']
    if image_file.filename == '':
        prompt_from_image = ""
    # Convert image to PIL or bytes for processing
    img = Image.open(image_file.stream)
    prompt_from_image = clip_model.get_prompt(img)
    prompt_list["prompt_from_image"] = prompt_from_image
    prompt_from_user = request.form.get('user_text')
    if not prompt_from_user:
        prompt_from_user = ""
    prompt_list["prompt_from_user"] = prompt_from_user
    flash = Flash()
    response = flash.getResponseFromFlash(prompt_list)
    message = response.text
    return jsonify({
        "prompt_from_image" : prompt_from_image, 
        "message": message
    })


if __name__ == '__main__':
    app.run(debug=True)