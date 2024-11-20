def forward_image(self, image):
    image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(query_embeds=query_tokens,
        encoder_hidden_states=image_embeds, encoder_attention_mask=
        image_atts, return_dict=True)
    return query_output.last_hidden_state, image_embeds
