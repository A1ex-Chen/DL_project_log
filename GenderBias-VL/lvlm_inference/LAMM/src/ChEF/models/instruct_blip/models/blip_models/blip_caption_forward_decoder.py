def forward_decoder(self, samples, image_embeds):
    raw_text = samples['text_input']
    text = self.tokenizer(raw_text, padding='longest', truncation=True,
        max_length=self.max_txt_len, return_tensors='pt').to(self.device)
    text.input_ids[:, 0] = self.tokenizer.bos_token_id
    decoder_targets = text.input_ids.masked_fill(text.input_ids == self.
        tokenizer.pad_token_id, -100)
    decoder_targets[:, :self.prompt_length] = -100
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self
        .device)
    decoder_output = self.text_decoder(input_ids=text.input_ids,
        attention_mask=text.attention_mask, encoder_hidden_states=
        image_embeds, encoder_attention_mask=image_atts, labels=
        decoder_targets, return_dict=True)
    return decoder_output, decoder_targets
