def get_metrics(run_results):
    n_all = 0
    n_top_1, n_top_3, n_top_5, mrr_5 = 0, 0, 0, 0
    n_acc_cand = 0
    n_candidates = 0
    for results in run_results:
        answer = results['ground_truths'][0]
        top_10_candidates = [result['phrase'] for result in results['result']]
        if isinstance(top_10_candidates[0], tuple):
            top_10_candidates = [pred for pred, _, _ in top_10_candidates]
        n_top_1 += 1.0 if answer in top_10_candidates[:1] else 0
        n_top_3 += 1.0 if answer in top_10_candidates[:3] else 0
        n_top_5 += 1.0 if answer in top_10_candidates[:5] else 0
        correct_pred_pos = top_10_candidates[:5].index(answer
            ) + 1 if answer in top_10_candidates[:5] else 0
        mrr_5 += 1.0 / correct_pred_pos if correct_pred_pos > 0 else 0
        n_all += 1
        n_acc_cand += 1 if results['included_in_candidates'] else 0
        n_candidates += results['number_of_candidates']
    if n_all > 0:
        metrics = {'Top@1': round(n_top_1 * 100 / n_all, 2), 'Top@3': round
            (n_top_3 * 100 / n_all, 2), 'Top@5': round(n_top_5 * 100 /
            n_all, 2), 'MRR@5': round(mrr_5 * 100 / n_all, 2),
            'Avg. acc for candidate extraction': float(n_acc_cand) / n_all,
            'Avg. number of candidates': float(n_candidates) / n_all,
            'Total count': n_all}
    else:
        metrics = {'error': 'n_all is 0'}
    return metrics
