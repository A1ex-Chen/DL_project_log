def test_eps_copa(cov_inter_score, inter_out_score, cov_num, iter_num,
    direct_match=False, normalization=True, use_cooccur=False, res_norm=
    True, ord=2, pxa_all_norm=False, eps=0.006029):
    unpacked_length = cov_inter_score.shape[0]
    causal_score_list = []
    for unpacked_idx in range(unpacked_length):
        cur_cov_inter = cov_inter_score[unpacked_idx]
        cur_inter_out = inter_out_score[unpacked_idx]
        causal_score = metrics.delta_bar(cur_cov_inter, cur_inter_out, eps=
            eps, direct_match=direct_match, normalization=normalization,
            use_cooccur=use_cooccur, res_norm=res_norm, ord=ord,
            pxa_all_norm=pxa_all_norm)
        causal_score_list.append(causal_score)
    return causal_score_list
