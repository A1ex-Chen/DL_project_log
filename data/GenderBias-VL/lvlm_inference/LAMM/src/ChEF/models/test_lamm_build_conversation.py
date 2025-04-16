def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, generate=True, **kwargs):
    if generate:
        prompts = self.generate_conversation_text([prompt], sys_msg=kwargs.
            get('sys_msg', None))
        prompt = prompts[0]
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
    else:
        conversation = []
        conversation.append({'from': 'human', 'value': prompt})
        fromgpt = batch_answers[idx]
        if CoT_answer_list is not None:
            fromgpt = CoT_answer_list[idx] + '\n' + fromgpt
        conversation.append({'from': 'gpt', 'value': fromgpt})
        prompt = conversation
    return prompt
