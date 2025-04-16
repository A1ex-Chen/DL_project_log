def get_state(self):
    logging.info('Saving state of the tokenizer')
    state = {'separator': self.separator, 'vocab_size': self.vocab_size,
        'tok2idx': self.tok2idx, 'idx2tok': self.idx2tok}
    return state
