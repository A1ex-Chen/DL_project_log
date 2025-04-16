def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    if isinstance(image_list, str):
        image_list = [image_list]
    prompt = wrap_question_with_default_conv(prompt, self.image_token_len *
        len(image_list))
    if CoT_answer_list is not None:
        prompt += CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += ' ' + batch_answers[idx]
    return prompt
