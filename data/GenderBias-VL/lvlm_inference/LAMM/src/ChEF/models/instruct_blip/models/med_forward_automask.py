def forward_automask(self, tokenized_text, visual_embeds, **kwargs):
    image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
        self.device)
    text = tokenized_text
    text_output = super().forward(text.input_ids, attention_mask=text.
        attention_mask, encoder_hidden_states=visual_embeds,
        encoder_attention_mask=image_atts, return_dict=True)
    return text_output
