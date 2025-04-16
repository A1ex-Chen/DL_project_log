def finalize_hypos(self, step: int, bbsz_idx, eos_scores, tokens, scores,
    finalized: List[List[Dict[str, Tensor]]], finished: List[bool],
    beam_size: int, attn: Optional[Tensor], src_lengths, max_len: int):
    """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
    assert bbsz_idx.numel() == eos_scores.numel()
    tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1:step + 2]
    tokens_clone[:, step] = self.eos
    attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2
        ] if attn is not None else None
    pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
    pos_scores[:, step] = eos_scores
    pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
    if self.normalize_scores:
        eos_scores /= (step + 1) ** self.len_penalty
    cum_unfin: List[int] = []
    prev = 0
    for f in finished:
        if f:
            prev += 1
        else:
            cum_unfin.append(prev)
    cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)
    unfin_idx = bbsz_idx // beam_size
    sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)
    seen = (sent << 32) + unfin_idx
    unique_seen: List[int] = torch.unique(seen).tolist()
    if self.match_source_len:
        condition = step > torch.index_select(src_lengths, 0, unfin_idx)
        eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores
            )
    sent_list: List[int] = sent.tolist()
    for i in range(bbsz_idx.size()[0]):
        if len(finalized[sent_list[i]]) < beam_size:
            if attn_clone is not None:
                hypo_attn = attn_clone[i]
            else:
                hypo_attn = torch.empty(0)
            finalized[sent_list[i]].append({'tokens': tokens_clone[i],
                'score': eos_scores[i], 'attention': hypo_attn, 'alignment':
                torch.empty(0), 'positional_scores': pos_scores[i]})
    newly_finished: List[int] = []
    for unique_s in unique_seen:
        unique_sent: int = unique_s >> 32
        unique_unfin_idx: int = unique_s - (unique_sent << 32)
        if not finished[unique_sent] and self.is_finished(step,
            unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size):
            finished[unique_sent] = True
            newly_finished.append(unique_unfin_idx)
    return newly_finished
