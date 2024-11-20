def encoder_decoder_mask(self, query_input: torch.Tensor, key_input: torch.
    Tensor) ->torch.Tensor:
    mask = torch.mul(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
    return mask.unsqueeze(-3)
