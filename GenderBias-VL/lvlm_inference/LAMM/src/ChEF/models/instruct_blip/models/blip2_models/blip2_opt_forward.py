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
    inputs_opt = self.opt_proj(query_output.last_hidden_state)
    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image
        .device)
    self.opt_tokenizer.padding_side = 'right'
    text = [(t + '\n') for t in samples['text_input']]
    opt_tokens = self.opt_tokenizer(text, return_tensors='pt', padding=
        'longest', truncation=True, max_length=self.max_txt_len).to(image.
        device)
    targets = opt_tokens.input_ids.masked_fill(opt_tokens.input_ids == self
        .opt_tokenizer.pad_token_id, -100)
    if self.prompt:
        targets[:, :self.prompt_length] = -100
    empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(image.
        device).fill_(-100)
    targets = torch.cat([empty_targets, targets], dim=1)
    inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.
        input_ids)
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
    with self.maybe_autocast():
        outputs = self.opt_model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, return_dict=True, labels=targets)
    loss = outputs.loss
    return {'loss': loss}
