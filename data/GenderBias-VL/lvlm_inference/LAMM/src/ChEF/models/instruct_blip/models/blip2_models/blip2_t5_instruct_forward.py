def forward(self, samples):
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(samples['text_input'], padding=
            'longest', truncation=True, max_length=self.max_txt_len,
            return_tensors='pt').to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
            dim=1)
        query_output = self.Qformer.bert(text_Qformer.input_ids,
            attention_mask=Qformer_atts, query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
    else:
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
    inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :
        query_tokens.size(1), :])
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.
        device)
    fs_embeds, fs_atts = None, None
    if self.few_shot_prob > 0 and 'few_shot_samples' in samples.keys():
        fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples[
            'few_shot_samples'])
    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(samples['text_input'], padding=
            'longest', truncation=True, max_length=self.max_txt_len,
            return_tensors='pt').to(image.device)
        output_tokens = self.t5_output_tokenizer(samples['text_output'],
            padding='longest', truncation=True, max_length=self.
            max_output_txt_len, return_tensors='pt').to(image.device)
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        targets = output_tokens.input_ids.masked_fill(output_tokens.
            input_ids == self.t5_tokenizer.pad_token_id, -100)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        if fs_embeds is not None:
            inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
            encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)
        outputs = self.t5_model(inputs_embeds=inputs_embeds, attention_mask
            =encoder_atts, decoder_attention_mask=output_tokens.
            attention_mask, return_dict=True, labels=targets)
        loss = outputs.loss
        return outputs, targets
