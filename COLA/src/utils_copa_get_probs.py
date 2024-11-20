def copa_get_probs(s, model, spacy_model, top_k=5):
    D, X, Dint, Y = s['text'], s['covariates'], s['interventions'], s['outcome'
        ]
    new_X = []
    for x in X:
        new_X.append(crop_sent(x, spacy_model=spacy_model))
    X = new_X
    baseln_probs = [model.get_temp(x, '', top_k=top_k) for x in X]
    tmp_probs = [[(model.get_temp(x, d, top_k=top_k) + baseln_probs[xidx]) for
        d in [D] + Dint] for xidx, x in enumerate(X)]
    tmp_y_probs = [model.get_temp(d, Y, top_k=top_k) for d in [D] + Dint]
    tmp_xy_probs = [model.get_temp(x, Y, top_k=top_k) for x in X]
    s['p_xd'] = tmp_probs
    s['p_dy'] = tmp_y_probs
    s['p_xy'] = tmp_xy_probs
    return s
