def __init__(self, model, tokenizer, batch_size, beam_size, min_length,
    max_length, alpha=0, block_repeating_trigram=True):
    """
        Attributes:
            mask_word_id: token id that corresponds to the mask
        """
    super(TransformerBeamSearch, self).__init__()
    self.model = model
    self.tokenizer = tokenizer
    self.start_token_id = tokenizer.start_token_id
    self.end_token_id = tokenizer.end_token_id
    self.pad_token_id = tokenizer.pad_token_id
    self.beam_size = beam_size
    self.min_length = min_length
    self.max_length = max_length
    self.block_repeating_trigram = block_repeating_trigram
    self.apply_length_penalty = False if alpha == 0 else True
    self.alpha = alpha
    self.hypotheses = [[] for _ in range(batch_size)]
    self.batch_offset = torch.arange(batch_size, dtype=torch.long)
    self.beam_offset = torch.arange(0, batch_size * self.beam_size, step=
        self.beam_size, dtype=torch.long)
    self.growing_beam = torch.full((batch_size * self.beam_size, 1), self.
        start_token_id, dtype=torch.long)
    self.topk_log_probabilities = torch.tensor([0.0] + [float('-inf')] * (
        self.beam_size - 1), dtype=torch.float).repeat(batch_size)
    self.results = {'prediction': [[] for _ in batch_size], 'scores': [[] for
        _ in batch_size]}
    self._step = 0
    self.is_done = False
