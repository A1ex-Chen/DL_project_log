def rouge_eval(hyps, refer, rouge_scorer):
    mean_score = rouge_scorer.get_scores(hyps, refer, avg=True)['rouge-1']['r']
    return mean_score
