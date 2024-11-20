def encoder_decoder_mask(self, query_input, key_input):
    mask = torch.mul(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
    return mask.unsqueeze(-3)
