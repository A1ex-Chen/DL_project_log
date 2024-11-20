def forward(self, encoder_inputs, encoder_inputs_mask):
    x = self.input_proj(encoder_inputs)
    max_positions = encoder_inputs.shape[1]
    input_positions = torch.arange(max_positions, device=encoder_inputs.device)
    seq_lens = encoder_inputs_mask.sum(-1)
    input_positions = torch.roll(input_positions.unsqueeze(0), tuple(
        seq_lens.tolist()), dims=0)
    x += self.position_encoding(input_positions)
    x = self.dropout_pre(x)
    input_shape = encoder_inputs.size()
    extended_attention_mask = self.get_extended_attention_mask(
        encoder_inputs_mask, input_shape)
    for lyr in self.encoders:
        x = lyr(x, extended_attention_mask)[0]
    x = self.layer_norm(x)
    return self.dropout_post(x), encoder_inputs_mask
