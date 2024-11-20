def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None,
    batch_answers=None, **kwargs):
    if isinstance(image_list, str):
        image_list = [image_list]
    format_list = []
    for image in image_list:
        if len(image) > 250:
            new_image_path = self.image_path_temp + image.split('/')[-1]
            shutil.copy(image, new_image_path)
            image = new_image_path
        format_list.append(dict(image=image))
    format_list.append(dict(text=prompt))
    query = self.tokenizer.from_list_format(format_list)
    raw_text, context_tokens = make_context(self.tokenizer, query=query,
        system='You are a helpful assistant.')
    if CoT_answer_list is not None:
        raw_text += CoT_answer_list[idx]
        context_tokens += self.tokenizer.encode(CoT_answer_list[idx],
            allowed_special=set(self.tokenizer.IMAGE_ST))
    if batch_answers is not None:
        raw_text += '\n ' + batch_answers[idx]
        context_tokens += self.tokenizer.encode('\n') + self.tokenizer.encode(
            ' ' + batch_answers[idx], allowed_special=set(self.tokenizer.
            IMAGE_ST))
    return raw_text, context_tokens
