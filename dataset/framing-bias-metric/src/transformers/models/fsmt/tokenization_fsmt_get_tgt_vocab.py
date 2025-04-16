def get_tgt_vocab(self):
    return dict(self.decoder, **self.added_tokens_decoder)
