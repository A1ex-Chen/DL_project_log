def to_openai_api_messages(self):
    """Convert the conversation to OpenAI chat completion format."""
    ret = [{'role': 'system', 'content': self.system}]
    for i, (_, msg) in enumerate(self.messages[self.offset:]):
        if i % 2 == 0:
            ret.append({'role': 'user', 'content': msg})
        elif msg is not None:
            ret.append({'role': 'assistant', 'content': msg})
    return ret
