def set_state(self, state):
    logging.info('Restoring state of the tokenizer')
    self.separator = state['separator']
    self.vocab_size = state['vocab_size']
    self.tok2idx = state['tok2idx']
    self.idx2tok = state['idx2tok']
