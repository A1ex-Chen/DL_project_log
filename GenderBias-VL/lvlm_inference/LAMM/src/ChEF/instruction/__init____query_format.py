def _query_format(self, prompt, question):
    if '{question}' in prompt:
        return prompt.format(question=question)
    assert question == '', f'Need question formatted in prompt, but "{prompt}" does not support.'
    return prompt
