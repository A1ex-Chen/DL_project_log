def forward(self, samples, match_head='itm'):
    image = samples['image']
    caption = samples['text_input']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    text = self.tokenizer(caption, truncation=True, max_length=self.
        max_txt_len, return_tensors='pt').to(image.device)
    if match_head == 'itm':
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        output_itm = self.Qformer.bert(text.input_ids, query_embeds=
            query_tokens, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        itm_embeddings = output_itm.last_hidden_state[:, :query_tokens.size
            (1), :]
        itm_logit = self.itm_head(itm_embeddings)
        itm_logit = itm_logit.mean(dim=1)
        return itm_logit
    elif match_head == 'itc':
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        image_feats = F.normalize(self.vision_proj(query_output.
            last_hidden_state), dim=-1)
        text_output = self.Qformer.bert(text.input_ids, attention_mask=text
            .attention_mask, return_dict=True)
        text_feat = F.normalize(self.text_proj(text_output.
            last_hidden_state[:, 0, :]), dim=-1)
        sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
        sim, _ = torch.max(sims, dim=1)
        return sim
