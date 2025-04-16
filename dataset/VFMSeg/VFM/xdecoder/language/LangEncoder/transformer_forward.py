def forward(self, input_ids, attention_mask=None):
    key_padding_mask = (attention_mask == 0 if not self.autogressive and 
        attention_mask is not None else None)
    x = self.token_embedding(input_ids)
    x = x + self.positional_embedding
    x = x.permute(1, 0, 2)
    for block in self.resblocks:
        x = block(x, key_padding_mask)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x)
    return {'last_hidden_state': x}
