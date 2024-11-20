def do_generate(self, images, prompt, max_new_tokens, **kwargs):
    tokenized = self.tokenizer([prompt])
    input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
    input_ids = torch_pad_sequence(input_ids, self.tokenizer.pad_token_id,
        padding_side='left')
    input_size = input_ids.shape[-1]
    attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')
    stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.
        tokenizer, input_size)
    output = self.model.generate(input_ids=input_ids.to(self.device),
        images=[images], attention_mask=attn_mask.to(self.device),
        temperature=0.7, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True, stopping_criteria
        =[stopping_criteria], repetition_penalty=1.1)
    output_id = output.sequences[0]
    response = self.tokenizer.decode(output_id[input_size:],
        skip_special_tokens=True)
    if response.count('###'):
        response = response[:response.index('###')]
    response = response.strip()
    return response
