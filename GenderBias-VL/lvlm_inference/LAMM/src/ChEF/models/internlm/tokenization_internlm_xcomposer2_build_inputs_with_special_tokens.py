def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []
    output = bos_token_ids + token_ids_0
    if token_ids_1 is not None:
        output = output + token_ids_1
    if self.add_eos_token:
        output = output + [self.eos_token_id]
    return output
