def find_match(source_sents, target_body):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    selected = []
    for sent in source_sents:
        scores = scorer.score(sent, target_body)
        r_score = scores['rouge1'].recall
        if r_score > 0.7:
            selected.append(sent)
    return selected
