def forward(self, samples):
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(query_embeds=query_tokens,
        encoder_hidden_states=image_embeds, encoder_attention_mask=
        image_atts, return_dict=True)
    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.
        device)
    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(samples['text_input'], padding=
            'longest', truncation=True, max_length=self.max_txt_len,
            return_tensors='pt').to(image.device)
        output_tokens = self.t5_tokenizer(samples['text_output'], padding=
            'longest', truncation=True, max_length=self.max_txt_len,
            return_tensors='pt').to(image.device)
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        targets = output_tokens.input_ids.masked_fill(output_tokens.
            input_ids == self.t5_tokenizer.pad_token_id, -100)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        outputs = self.t5_model(inputs_embeds=inputs_embeds, attention_mask
            =encoder_atts, decoder_attention_mask=output_tokens.
            attention_mask, return_dict=True, labels=targets)
        loss = outputs.loss
        return {'loss': loss}
