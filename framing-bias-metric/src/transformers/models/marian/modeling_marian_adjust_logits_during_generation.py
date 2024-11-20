def adjust_logits_during_generation(self, logits, cur_len, max_length):
    logits[:, self.config.pad_token_id] = float('-inf')
    if cur_len == max_length - 1 and self.config.eos_token_id is not None:
        self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
    return logits
