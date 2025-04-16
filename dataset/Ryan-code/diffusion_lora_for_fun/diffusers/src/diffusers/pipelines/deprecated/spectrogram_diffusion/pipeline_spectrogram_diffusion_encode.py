def encode(self, input_tokens, continuous_inputs, continuous_mask):
    tokens_mask = input_tokens > 0
    tokens_encoded, tokens_mask = self.notes_encoder(encoder_input_tokens=
        input_tokens, encoder_inputs_mask=tokens_mask)
    continuous_encoded, continuous_mask = self.continuous_encoder(
        encoder_inputs=continuous_inputs, encoder_inputs_mask=continuous_mask)
    return [(tokens_encoded, tokens_mask), (continuous_encoded,
        continuous_mask)]
