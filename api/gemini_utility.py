import google.generativeai as genai
genai.configure(api_key='AIzaSyD5ayo_ZGg31JxjIuCT9nf-ZKZwuHgHUbQ')

#class dedicated to get response from gemini-flash
class Flash:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    def getResponseFromFlash(self,prompt_list):
        prompt_from_image = prompt_list.get('prompt_from_image')
        prompt_from_user = prompt_list.get('prompt_from_image') 
        if prompt_from_image != "": 
            prompt = prompt_from_image
        else:
            prompt = "tell me about your self"
        response = self.model.generate_content(prompt)
        return(response)