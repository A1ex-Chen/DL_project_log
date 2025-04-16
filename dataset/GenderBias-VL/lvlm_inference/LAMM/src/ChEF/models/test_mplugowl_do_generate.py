def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
    inputs = self.processor(text=[prompt])
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    inputs['pixel_values'] = image_list
    generate_kwargs = {'do_sample': False, 'top_k': 5, 'max_length':
        max_new_tokens}
    with torch.no_grad():
        res = self.model.generate(**inputs, **generate_kwargs)
    outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for
        output in res.tolist()]
    return outputs[0]
