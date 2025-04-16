def do_generate(self, image_list: list, ds: SingleImageInteractive,
    max_new_tokens, **kwargs):
    model_inputs = ds.to_model_input()
    text = model_inputs['input_text']
    images = model_inputs['images'].to(dtype=torch.float16, device=self.device)
    input_dict = self.tokenizer([text], padding='longest', return_length=
        True, add_special_tokens=False, return_tensors='pt').to(self.device)
    input_ids = input_dict['input_ids']
    attention_mask = input_dict['attention_mask']
    output_ids = self.model.generate(images=images, input_ids=input_ids,
        attention_mask=attention_mask, max_new_tokens=max_new_tokens, **
        self.gen_kwargs)
    input_token_len = input_ids.shape[-1]
    output = self.tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
    return output.split('</s>')[0]
