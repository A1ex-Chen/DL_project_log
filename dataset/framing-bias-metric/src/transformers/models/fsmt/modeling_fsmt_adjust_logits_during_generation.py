def adjust_logits_during_generation(self, logits, cur_len, max_length):
    if cur_len == max_length - 1 and self.config.eos_token_id is not None:
        self._force_token_ids_generation(logits, self.config.eos_token_id)
    return logits
