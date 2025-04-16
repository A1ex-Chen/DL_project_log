def forward(self, samples, match_head='itm'):
    image = samples['image']
    caption = samples['text_input']
    image_embeds = self.visual_encoder.forward_features(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    text = self.tokenizer(caption, padding='longest', truncation=True,
        max_length=self.max_txt_len, return_tensors='pt').to(image.device)
    if match_head == 'itm':
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        output = self.text_encoder(encoder_input_ids, attention_mask=text.
            attention_mask, encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts, return_dict=True)
        itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
        return itm_output
    elif match_head == 'itc':
        text_output = self.text_encoder(text.input_ids, attention_mask=text
            .attention_mask, return_dict=True, mode='text')
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]),
            dim=-1)
        text_feat = F.normalize(self.text_proj(text_output.
            last_hidden_state[:, 0, :]), dim=-1)
        sim = image_feat @ text_feat.t()
        return sim
