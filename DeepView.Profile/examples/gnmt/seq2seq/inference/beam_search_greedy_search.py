def greedy_search(self, batch_size, initial_input, initial_context=None):
    """
        Greedy decoder.

        :param batch_size: decoder batch size
        :param initial_input: initial input, usually tensor of BOS tokens
        :param initial_context: initial context, usually [encoder_context,
            src_seq_lengths, None]

        returns: (translation, lengths, counter)
            translation: (batch_size, max_seq_len) - indices of target tokens
            lengths: (batch_size) - lengths of generated translations
            counter: number of iterations of the decoding loop
        """
    max_seq_len = self.max_seq_len
    translation = torch.zeros(batch_size, max_seq_len, dtype=torch.int64)
    lengths = torch.ones(batch_size, dtype=torch.int64)
    active = torch.arange(0, batch_size, dtype=torch.int64)
    base_mask = torch.arange(0, batch_size, dtype=torch.int64)
    if self.cuda:
        translation = translation.cuda()
        lengths = lengths.cuda()
        active = active.cuda()
        base_mask = base_mask.cuda()
    translation[:, 0] = BOS
    words, context = initial_input, initial_context
    if self.batch_first:
        word_view = -1, 1
        ctx_batch_dim = 0
    else:
        word_view = 1, -1
        ctx_batch_dim = 1
    counter = 0
    for idx in range(1, max_seq_len):
        if not len(active):
            break
        counter += 1
        words = words.view(word_view)
        output = self.model.generate(words, context, 1)
        words, logprobs, attn, context = output
        words = words.view(-1)
        translation[active, idx] = words
        lengths[active] += 1
        terminating = words == EOS
        if terminating.any():
            not_terminating = ~terminating
            mask = base_mask[:len(active)]
            mask = mask.masked_select(not_terminating)
            active = active.masked_select(not_terminating)
            words = words[mask]
            context[0] = context[0].index_select(ctx_batch_dim, mask)
            context[1] = context[1].index_select(0, mask)
            context[2] = context[2].index_select(1, mask)
    return translation, lengths, counter
