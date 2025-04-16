def adjust_logits_during_generation(self, logits, cur_len, max_length):
    """Never predict pad_token_id. Predict </s> when max_length is reached."""
    vocab_range = tf.constant(range(self.config.vocab_size))
    logits = tf.where(vocab_range == self.config.pad_token_id,
        LARGE_NEGATIVE, logits)
    if cur_len == max_length - 1:
        logits = tf.where(vocab_range != self.config.eos_token_id,
            LARGE_NEGATIVE, logits)
    return logits
