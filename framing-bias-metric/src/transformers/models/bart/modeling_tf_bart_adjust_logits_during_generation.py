def adjust_logits_during_generation(self, logits, cur_len, max_length):
    if cur_len == 1 and self.config.force_bos_token_to_be_generated:
        vocab_range = tf.constant(range(self.config.vocab_size))
        return tf.where(vocab_range != self.config.bos_token_id,
            LARGE_NEGATIVE, logits)
    elif cur_len == max_length - 1:
        vocab_range = tf.constant(range(self.config.vocab_size))
        return tf.where(vocab_range != self.config.eos_token_id,
            LARGE_NEGATIVE, logits)
    else:
        return logits
