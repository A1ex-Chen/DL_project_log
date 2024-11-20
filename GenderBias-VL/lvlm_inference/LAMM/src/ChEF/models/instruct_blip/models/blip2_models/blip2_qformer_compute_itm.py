def compute_itm(self, image_inputs, text_ids, text_atts):
    image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
        image_inputs.device)
    query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        image_inputs.device)
    attention_mask = torch.cat([query_atts, text_atts], dim=1)
    output_itm = self.Qformer.bert(text_ids, query_embeds=query_tokens,
        attention_mask=attention_mask, encoder_hidden_states=image_inputs,
        encoder_attention_mask=image_atts, return_dict=True)
    vl_embeddings = output_itm.last_hidden_state[:, :query_tokens.size(1), :]
    itm_logit = self.itm_head(vl_embeddings)
    itm_logit = itm_logit[:, :, 1].mean(dim=1)
    return itm_logit
