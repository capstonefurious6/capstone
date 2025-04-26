import google.generativeai as genai

# def query_gemini(api_key, query_text):
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel(model_name="gemini-1.5-flash")
#     response = model.generate_content(query_text)
#     return response.text

def query_gemini(api_key, query_text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(query_text)
    return "".join(part.text for part in response.parts)