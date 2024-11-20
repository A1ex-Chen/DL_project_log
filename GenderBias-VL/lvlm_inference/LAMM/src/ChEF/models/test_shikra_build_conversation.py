def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    ds = prepare_interactive(model_args, self.preprocessor)
    ds.set_image(self.build_input_image(image_list))
    ds.append_message(role=ds.roles[0], message=prompt, boxes=[], boxes_seq=[])
    user_msg = ''
    if CoT_answer_list is not None:
        user_msg += CoT_answer_list[idx]
    if batch_answers is not None:
        user_msg += '\n ' + batch_answers[idx]
    if user_msg != '':
        ds.append_message(role=ds.roles[1], message=user_msg, boxes=[],
            boxes_seq=[])
    return ds
