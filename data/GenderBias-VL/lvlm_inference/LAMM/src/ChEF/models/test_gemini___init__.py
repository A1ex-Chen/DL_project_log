def __init__(self, api_key='', gemini_name='gemini-pro-vision',
    safety_block_none=False, **kwargs) ->None:
    genai.configure(api_key=api_key)
    self.model = genai.GenerativeModel(gemini_name)
    self.gemini_name = gemini_name
    self.safety_block_none = safety_block_none
