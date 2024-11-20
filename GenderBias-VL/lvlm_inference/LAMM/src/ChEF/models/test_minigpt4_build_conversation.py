def build_conversation(self, idx, image_tensor_list, prompt,
    CoT_answer_list=None, batch_answers=None, **kwargs):
    conv = CONV_VISION.copy()
    conv.append_message(conv.roles[0], ' '.join(['<Img><ImageHere></Img>'] *
        len(image_tensor_list)))
    self.chat.ask(prompt, conv)
    conv.append_message(conv.roles[1], None)
    query = conv.get_prompt()
    if CoT_answer_list is not None:
        query += CoT_answer_list[idx]
    if batch_answers is not None:
        query += ' ' + batch_answers[idx]
    return query
