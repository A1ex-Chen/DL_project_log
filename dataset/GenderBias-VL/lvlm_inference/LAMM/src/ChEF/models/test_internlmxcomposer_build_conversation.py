def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    prompt = f'<ImageHere>{prompt}'
    prompt = self.model.build_inputs(prompt, [])
    if CoT_answer_list is not None:
        prompt += CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += ' ' + batch_answers[idx]
    return prompt
