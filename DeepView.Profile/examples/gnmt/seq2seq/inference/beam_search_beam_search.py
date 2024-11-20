def beam_search(self, batch_size, initial_input, initial_context=None):
    """
        Beam search decoder.

        :param batch_size: decoder batch size
        :param initial_input: initial input, usually tensor of BOS tokens
        :param initial_context: initial context, usually [encoder_context,
            src_seq_lengths, None]

        returns: (translation, lengths, counter)
            translation: (batch_size, max_seq_len) - indices of target tokens
            lengths: (batch_size) - lengths of generated translations
            counter: number of iterations of the decoding loop
        """
    beam_size = self.beam_size
    norm_const = self.len_norm_const
    norm_factor = self.len_norm_factor
    max_seq_len = self.max_seq_len
    cov_penalty_factor = self.cov_penalty_factor
    translation = torch.zeros(batch_size * beam_size, max_seq_len, dtype=
        torch.int64)
    lengths = torch.ones(batch_size * beam_size, dtype=torch.int64)
    scores = torch.zeros(batch_size * beam_size, dtype=torch.float32)
    active = torch.arange(0, batch_size * beam_size, dtype=torch.int64)
    base_mask = torch.arange(0, batch_size * beam_size, dtype=torch.int64)
    global_offset = torch.arange(0, batch_size * beam_size, beam_size,
        dtype=torch.int64)
    eos_beam_fill = torch.tensor([0] + (beam_size - 1) * [float('-inf')])
    if self.cuda:
        translation = translation.cuda()
        lengths = lengths.cuda()
        active = active.cuda()
        base_mask = base_mask.cuda()
        scores = scores.cuda()
        global_offset = global_offset.cuda()
        eos_beam_fill = eos_beam_fill.cuda()
    translation[:, 0] = BOS
    words, context = initial_input, initial_context
    if self.batch_first:
        word_view = -1, 1
        ctx_batch_dim = 0
        attn_query_dim = 1
    else:
        word_view = 1, -1
        ctx_batch_dim = 1
        attn_query_dim = 0
    if self.batch_first:
        _, seq, feature = context[0].shape
        context[0].unsqueeze_(1)
        context[0] = context[0].expand(-1, beam_size, -1, -1)
        context[0] = context[0].contiguous().view(batch_size * beam_size,
            seq, feature)
    else:
        seq, _, feature = context[0].shape
        context[0].unsqueeze_(2)
        context[0] = context[0].expand(-1, -1, beam_size, -1)
        context[0] = context[0].contiguous().view(seq, batch_size *
            beam_size, feature)
    context[1].unsqueeze_(1)
    context[1] = context[1].expand(-1, beam_size)
    context[1] = context[1].contiguous().view(batch_size * beam_size)
    accu_attn_scores = torch.zeros(batch_size * beam_size, seq)
    if self.cuda:
        accu_attn_scores = accu_attn_scores.cuda()
    counter = 0
    for idx in range(1, self.max_seq_len):
        if not len(active):
            break
        counter += 1
        eos_mask = words == EOS
        eos_mask = eos_mask.view(-1, beam_size)
        terminating, _ = eos_mask.min(dim=1)
        lengths[active[~eos_mask.view(-1)]] += 1
        output = self.model.generate(words, context, beam_size)
        words, logprobs, attn, context = output
        attn = attn.float().squeeze(attn_query_dim)
        attn = attn.masked_fill(eos_mask.view(-1).unsqueeze(1), 0)
        accu_attn_scores[active] += attn
        words = words.view(-1, beam_size, beam_size)
        words = words.masked_fill(eos_mask.unsqueeze(2), EOS)
        logprobs = logprobs.float().view(-1, beam_size, beam_size)
        if eos_mask.any():
            logprobs[eos_mask] = eos_beam_fill
        active_scores = scores[active].view(-1, beam_size)
        new_scores = active_scores.unsqueeze(2) + logprobs
        if idx == 1:
            new_scores[:, 1:, :].fill_(float('-inf'))
        new_scores = new_scores.view(-1, beam_size * beam_size)
        _, index = new_scores.topk(beam_size, dim=1)
        source_beam = index / beam_size
        new_scores = new_scores.view(-1, beam_size * beam_size)
        best_scores = torch.gather(new_scores, 1, index)
        scores[active] = best_scores.view(-1)
        words = words.view(-1, beam_size * beam_size)
        words = torch.gather(words, 1, index)
        words = words.view(word_view)
        offset = global_offset[:source_beam.shape[0]]
        source_beam += offset.unsqueeze(1)
        translation[active, :] = translation[active[source_beam.view(-1)], :]
        translation[active, idx] = words.view(-1)
        lengths[active] = lengths[active[source_beam.view(-1)]]
        context[2] = context[2].index_select(1, source_beam.view(-1))
        if terminating.any():
            not_terminating = ~terminating
            not_terminating = not_terminating.unsqueeze(1)
            not_terminating = not_terminating.expand(-1, beam_size).contiguous(
                )
            normalization_mask = active.view(-1, beam_size)[terminating]
            norm = lengths[normalization_mask].float()
            norm = (norm_const + norm) / (norm_const + 1.0)
            norm = norm ** norm_factor
            scores[normalization_mask] /= norm
            penalty = accu_attn_scores[normalization_mask]
            penalty = penalty.clamp(0, 1)
            penalty = penalty.log()
            penalty[penalty == float('-inf')] = 0
            penalty = penalty.sum(dim=-1)
            scores[normalization_mask] += cov_penalty_factor * penalty
            mask = base_mask[:len(active)]
            mask = mask.masked_select(not_terminating.view(-1))
            words = words.index_select(ctx_batch_dim, mask)
            context[0] = context[0].index_select(ctx_batch_dim, mask)
            context[1] = context[1].index_select(0, mask)
            context[2] = context[2].index_select(1, mask)
            active = active.masked_select(not_terminating.view(-1))
    scores = scores.view(batch_size, beam_size)
    _, idx = scores.max(dim=1)
    translation = translation[idx + global_offset, :]
    lengths = lengths[idx + global_offset]
    return translation, lengths, counter
