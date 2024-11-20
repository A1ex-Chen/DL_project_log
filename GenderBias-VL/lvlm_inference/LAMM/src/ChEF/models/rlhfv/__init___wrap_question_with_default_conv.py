def wrap_question_with_default_conv(question_text, image_token_len):
    question_text = expand_question_into_multimodal(question_text,
        image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        DEFAULT_IMAGE_PATCH_TOKEN)
    conv = conv_templates['default'].copy()
    conv.messages = []
    conv.sep = '\n###'
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt
