@classmethod
def _parse_response_with_negative(cls, text):
    if not text:
        raise ValueError('LLM response is empty')
    if cls.objects_text in text:
        text = text.split(cls.objects_text)[1]
    text_split = text.split(cls.bg_prompt_text_no_trailing_space)
    if len(text_split) == 2:
        gen_boxes, text_rem = text_split
    else:
        raise ValueError(f'LLM response is incomplete: {text}')
    text_split = text_rem.split(cls.neg_prompt_text_no_trailing_space)
    if len(text_split) == 2:
        bg_prompt, neg_prompt = text_split
    else:
        raise ValueError(f'LLM response is incomplete: {text}')
    try:
        gen_boxes = ast.literal_eval(gen_boxes)
    except SyntaxError as e:
        if 'No objects' in gen_boxes or gen_boxes.strip() == '':
            gen_boxes = []
        else:
            raise e
    bg_prompt = bg_prompt.strip()
    neg_prompt = neg_prompt.strip()
    if neg_prompt == 'None':
        neg_prompt = ''
    return gen_boxes, bg_prompt, neg_prompt
