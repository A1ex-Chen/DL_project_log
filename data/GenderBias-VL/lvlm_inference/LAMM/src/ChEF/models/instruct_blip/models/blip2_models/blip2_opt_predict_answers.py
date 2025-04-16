def predict_answers(self, samples, num_beams=5, inference_method='generate',
    max_len=10, min_len=1, num_ans_candidates=128, answer_list=None, prompt
    ='', length_penalty=0, **kwargs):
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device)
        if isinstance(samples['text_input'], str):
            samples['text_input'] = [samples['text_input']]
        if prompt:
            text_input = [prompt.format(question) for question in samples[
                'text_input']]
        else:
            text_input = samples['text_input']
        self.opt_tokenizer.padding_side = 'left'
        opt_tokens = self.opt_tokenizer(text_input, return_tensors='pt',
            padding='longest', truncation=True, max_length=self.max_txt_len
            ).to(image.device)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1
            )
        inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        outputs = self.opt_model.generate(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, do_sample=False, num_beams=
            num_beams, max_new_tokens=max_len, min_length=min_len,
            eos_token_id=self.eos_token_id, length_penalty=length_penalty)
        output_text = self.opt_tokenizer.batch_decode(outputs,
            skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
    if self._apply_lemmatizer or 'apply_lemmatizer' in samples.keys(
        ) and samples['apply_lemmatizer']:
        output_text = self._lemmatize(output_text)
    return output_text
