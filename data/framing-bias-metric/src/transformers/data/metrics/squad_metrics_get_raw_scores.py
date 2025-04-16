def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer['text'] for answer in example.answers if
            normalize_answer(answer['text'])]
        if not gold_answers:
            gold_answers = ['']
        if qas_id not in preds:
            print('Missing prediction for %s' % qas_id)
            continue
        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in
            gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers
            )
    return exact_scores, f1_scores
