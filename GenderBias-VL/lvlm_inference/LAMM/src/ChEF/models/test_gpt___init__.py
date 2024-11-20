def __init__(self, api_key='', gpt_name='gpt-4-vision-preview', **kwargs
    ) ->None:
    self.client = OpenAI(api_key=api_key)
    self.gpt_name = gpt_name
