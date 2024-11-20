def forward(self, input_ids: torch.Tensor, prefix_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.
    Tensor]=None):
    """
        Args:
            input_ids (`torch.Tensor` of shape `(N, max_seq_len)`):
                Text tokens to use for inference.
            prefix_embeds (`torch.Tensor` of shape `(N, prefix_length, 768)`):
                Prefix embedding to preprend to the embedded tokens.
            attention_mask (`torch.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                Attention mask for the prefix embedding.
            labels (`torch.Tensor`, *optional*):
                Labels to use for language modeling.
        """
    embedding_text = self.transformer.transformer.wte(input_ids)
    hidden = self.encode_prefix(prefix_embeds)
    prefix_embeds = self.decode_prefix(hidden)
    embedding_cat = torch.cat((prefix_embeds, embedding_text), dim=1)
    if labels is not None:
        dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device
            )
        labels = torch.cat((dummy_token, input_ids), dim=1)
    out = self.transformer(inputs_embeds=embedding_cat, labels=labels,
        attention_mask=attention_mask)
    if self.prefix_hidden_dim is not None:
        return out, hidden
    else:
        return out
