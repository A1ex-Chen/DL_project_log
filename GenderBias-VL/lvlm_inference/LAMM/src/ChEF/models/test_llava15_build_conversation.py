def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    conv = self.conv.copy()
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if CoT_answer_list is not None:
        prompt += CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += '\n' + batch_answers[idx]
    return prompt
