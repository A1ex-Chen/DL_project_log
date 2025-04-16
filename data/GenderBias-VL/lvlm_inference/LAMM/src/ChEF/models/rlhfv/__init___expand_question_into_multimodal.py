def expand_question_into_multimodal(question_text, image_token_len,
    im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text:
        question_text = question_text.replace('<image>', '')
    question_text = (question_text + '\n' + im_st_token + im_patch_token *
        image_token_len + im_ed_token)
    return question_text
