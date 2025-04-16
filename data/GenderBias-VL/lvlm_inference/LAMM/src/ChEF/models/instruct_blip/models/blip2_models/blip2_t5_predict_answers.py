def predict_answers(self, samples, num_beams=5, inference_method='generate',
    max_len=10, min_len=1, num_ans_candidates=128, answer_list=None, prompt
    ='', length_penalty=-1, **kwargs):
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(query_embeds=query_tokens,
        encoder_hidden_states=image_embeds, encoder_attention_mask=
        image_atts, return_dict=True)
    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.
        device)
    if isinstance(samples['text_input'], str):
        samples['text_input'] = [samples['text_input']]
    if prompt:
        text_input = [prompt.format(question) for question in samples[
            'text_input']]
    else:
        text_input = samples['text_input']
    input_tokens = self.t5_tokenizer(text_input, padding='longest',
        return_tensors='pt').to(image.device)
    encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
    with self.maybe_autocast(dtype=torch.bfloat16):
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        outputs = self.t5_model.generate(inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts, do_sample=False, num_beams=
            num_beams, max_new_tokens=max_len, min_length=min_len,
            length_penalty=length_penalty)
        output_text = self.t5_tokenizer.batch_decode(outputs,
            skip_special_tokens=True)
    if self._apply_lemmatizer:
        output_text = self._lemmatize(output_text)
    return output_text
