def generate_from_encoder(self, tokenized_prompt, visual_embeds,
    sep_token_id, pad_token_id, use_nucleus_sampling=False, num_beams=3,
    max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0, **kwargs):
    if not use_nucleus_sampling:
        num_beams = num_beams
        visual_embeds = visual_embeds.repeat_interleave(num_beams, dim=0)
    image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
        self.device)
    model_kwargs = {'encoder_hidden_states': visual_embeds,
        'encoder_attention_mask': image_atts}
    if use_nucleus_sampling:
        outputs = self.generate(input_ids=tokenized_prompt.input_ids,
            max_length=max_length, min_length=min_length, do_sample=True,
            top_p=top_p, num_return_sequences=1, eos_token_id=sep_token_id,
            pad_token_id=pad_token_id, repetition_penalty=1.1, **model_kwargs)
    else:
        outputs = self.generate(input_ids=tokenized_prompt.input_ids,
            max_length=max_length, min_length=min_length, num_beams=
            num_beams, eos_token_id=sep_token_id, pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty, **model_kwargs)
    return outputs
