def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    prefix = ('[image]<image><tab><grounding>' if self.if_grounding else
        '[image]<image><tab>')
    prompt = ' '.join([prefix] * len(image_list)) + prompt
    if CoT_answer_list is not None:
        prompt += ' ' + CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += '\n' + batch_answers[idx]
    return prompt
