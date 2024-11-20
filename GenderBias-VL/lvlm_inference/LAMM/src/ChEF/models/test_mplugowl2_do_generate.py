def do_generate(self, input_image_list: list, input_prompt: str, max_new_tokens
    ):
    input_ids = tokenizer_image_token(input_prompt, self.tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
    conv = conv_templates['mplug_owl2'].copy()
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer,
        input_ids)
    streamer = TextStreamer(self.tokenizer, skip_prompt=True,
        skip_special_tokens=True)
    with torch.inference_mode():
        output_ids = self.model.generate(input_ids, images=input_image_list,
            do_sample=False, temperature=0.7, max_new_tokens=max_new_tokens,
            streamer=streamer, use_cache=True, stopping_criteria=[
            stopping_criteria])
    outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs
