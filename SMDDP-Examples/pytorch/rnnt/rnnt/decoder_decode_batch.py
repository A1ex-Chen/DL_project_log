def decode_batch(self, x, out_lens):
    """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """

    def next_multiple_of_eight(x):
        return math.ceil(x / 8) * 8
    assert self.max_symbol_per_sample is not None, 'max_symbol_per_sample needs to be specified in order to use batch_eval'
    with torch.no_grad():
        B = x.size(1)
        if B < 128:
            B_pad = next_multiple_of_eight(B)
            x = torch.nn.functional.pad(x, (0, 0, 0, B_pad - B))
        logits, out_lens = self.model.encode(x, out_lens)
        logits = logits[:B]
        output = []
        if self.batch_eval_mode == 'cg':
            output = self._greedy_decode_batch_replay(logits, out_lens)
        elif self.batch_eval_mode == 'cg_unroll_pipeline':
            output = self._greedy_decode_batch_replay_pipelined(logits,
                out_lens)
        elif self.batch_eval_mode == 'no_cg':
            output = self._greedy_decode_batch(logits, out_lens)
    return output
