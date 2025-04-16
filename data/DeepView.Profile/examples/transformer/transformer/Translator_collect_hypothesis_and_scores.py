def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]
        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in
            tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
