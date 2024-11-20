def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None
    ) ->List[int]:
    """Build model inputs from a sequence by appending eos_token_id."""
    if token_ids_1 is None:
        return token_ids_0 + [self.eos_token_id]
    return token_ids_0 + token_ids_1 + [self.eos_token_id]
