def do_generate(self, image_list, prompt: tuple, max_new_tokens, **kwargs):
    raw_text, context_tokens = prompt
    input_ids = torch.tensor([context_tokens]).to(self.device)
    outputs = self.model.generate(input_ids, stop_words_ids=self.
        stop_words_ids, return_dict_in_generate=False, generation_config=
        self.model.generation_config, do_sample=False, **kwargs)
    response = decode_tokens(outputs[0], self.tokenizer, raw_text_len=len(
        raw_text), context_length=len(context_tokens), chat_format=self.
        model.generation_config.chat_format, verbose=False, errors='replace')
    return response
