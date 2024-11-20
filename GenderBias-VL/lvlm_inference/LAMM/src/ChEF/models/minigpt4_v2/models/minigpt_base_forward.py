def forward(self, samples, reduction='mean'):
    cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = (self
        .preparing_embedding(samples))
    inputs_embeds, attention_mask, input_lens = self.concat_emb_input_output(
        cond_embeds, cond_atts, regress_embeds, regress_atts)
    bos = torch.ones_like(part_targets[:, :1]
        ) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.embed_tokens(bos)
    bos_atts = cond_atts[:, :1]
    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([bos_atts, attention_mask], dim=1)
    targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
        dtype=torch.long).to(self.device).fill_(-100)
    for i, target in enumerate(part_targets):
        targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target
    with self.maybe_autocast():
        outputs = self.llama_model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, return_dict=True, labels=targets,
            reduction=reduction)
    loss = outputs.loss
    return {'loss': loss}
