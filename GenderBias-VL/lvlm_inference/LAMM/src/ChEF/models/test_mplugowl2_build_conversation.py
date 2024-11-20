def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    conv = conv_templates['mplug_owl2'].copy()
    lenimg = 1 if isinstance(image_list, str) else len(image_list)
    inp = DEFAULT_IMAGE_TOKEN * lenimg + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if CoT_answer_list is not None:
        prompt += CoT_answer_list[idx]
    if batch_answers is not None:
        prompt += ' ' + batch_answers[idx]
    return prompt
