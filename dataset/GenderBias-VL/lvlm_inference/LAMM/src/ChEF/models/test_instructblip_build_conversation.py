def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    if CoT_answer_list is not None:
        prompt += '\n' + CoT_answer_list[idx]
    return prompt
