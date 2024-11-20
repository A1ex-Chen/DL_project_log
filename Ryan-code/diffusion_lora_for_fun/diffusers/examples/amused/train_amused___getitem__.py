def __getitem__(self, index):
    item = self.hf_dataset[index]
    rv = process_image(item[self.image_key], self.size)
    prompt = item[self.prompt_key]
    if self.prompt_prefix is not None:
        prompt = self.prompt_prefix + prompt
    rv['prompt_input_ids'] = tokenize_prompt(self.tokenizer, prompt)[0]
    return rv
