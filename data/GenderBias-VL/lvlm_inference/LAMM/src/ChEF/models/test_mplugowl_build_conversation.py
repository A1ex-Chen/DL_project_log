def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    lenimg = 1 if isinstance(image_list, str) else len(image_list)
    prompt = prompt_template_multi + '\n'.join(['Human: <image>'] * lenimg) + f"""
Human: {prompt}
AI:"""
    if CoT_answer_list is not None:
        prompt += CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += ' ' + batch_answers[idx]
    return prompt
