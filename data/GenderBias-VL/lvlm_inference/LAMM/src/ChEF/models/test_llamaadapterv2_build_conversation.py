def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, generate=True, **kwargs):
    prompt = llama_adapter_v2.format_prompt(prompt)
    if CoT_answer_list is not None:
        prompt += ' ' + CoT_answer_list[idx] + '\n'
    if generate:
        return prompt
    else:
        return prompt, batch_answers[idx]
