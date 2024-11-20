def adjust_logits_during_generation(self, logits, cur_len, max_length):
    if cur_len == 1 and self.config.force_bos_token_to_be_generated:
        self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
    elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
        self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
    return logits
