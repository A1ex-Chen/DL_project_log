def glt_get_probs(s, model, spacy_model, top_k=5):
    D, Y, X, Dint = s[['text', 'outcome', 'covariates', 'interventions']]
    if 'covariates_cleaned' in s:
        print(f"{s['index']}: using cleaned X")
        X = s['covariates_cleaned']
    else:
        X = [crop_sent(x, spacy_model=spacy_model) for x in X]
    baseln_probs = [model.get_temp(x, '', top_k=top_k) for x in X]
    tmp_probs = [[(model.get_temp(x, d, top_k=top_k) + baseln_probs[xidx]) for
        d in [D] + Dint] for xidx, x in enumerate(X)]
    tmp_y_probs = [model.get_temp(d, Y, top_k=top_k) for d in [D] + Dint]
    tmp_xy_probs = [model.get_temp(x, Y, top_k=top_k) for x in X]
    s['p_xd'] = tmp_probs
    s['p_dy'] = tmp_y_probs
    s['p_xy'] = tmp_xy_probs
    return s
