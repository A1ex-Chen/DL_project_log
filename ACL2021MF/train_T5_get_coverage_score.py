def get_coverage_score(gt_concepts, pred):
    covs = []
    total_cs, match_cs = 0, 0
    for cs, p in zip(gt_concepts, pred):
        p = p.lower()
        if p.endswith('.'):
            p = p[:-1]
            p = p.strip()
        cs = set(cs)
        lemmas = set()
        for token in nlp(p):
            lemmas.add(token.lemma_)
        match_cs += len(lemmas & cs)
        total_cs += len(cs)
        cov = len(lemmas & cs) / len(cs)
        covs.append(cov)
    return 100 * sum(covs) / len(covs), 100 * match_cs / total_cs
