def forward(self, encoder_input_tokens, encoder_inputs_mask):
    x = self.token_embedder(encoder_input_tokens)
    seq_length = encoder_input_tokens.shape[1]
    inputs_positions = torch.arange(seq_length, device=encoder_input_tokens
        .device)
    x += self.position_encoding(inputs_positions)
    x = self.dropout_pre(x)
    input_shape = encoder_input_tokens.size()
    extended_attention_mask = self.get_extended_attention_mask(
        encoder_inputs_mask, input_shape)
    for lyr in self.encoders:
        x = lyr(x, extended_attention_mask)[0]
    x = self.layer_norm(x)
    return self.dropout_post(x), encoder_inputs_mask
