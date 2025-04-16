def decode(self, x, out_lens):
    if self.batch_eval_mode is None:
        return self.decode_reference(self.model, x, out_lens)
    else:
        return self.decode_batch(x, out_lens)
