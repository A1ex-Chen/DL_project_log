def forward(self, input_encoder, input_enc_len, input_decoder):
    context = self.encode(input_encoder, input_enc_len)
    context = context, input_enc_len, None
    output, _, _ = self.decode(input_decoder, context)
    return output
