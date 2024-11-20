def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    if token_ids_1 is None:
        return output
    return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
