def squad_evaluate(examples, preds, no_answer_probs=None,
    no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for
        example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in
        qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer
        .items() if not has_answer]
    if no_answer_probs is None:
        no_answer_probs = {k: (0.0) for k in preds}
    exact, f1 = get_raw_scores(examples, preds)
    exact_threshold = apply_no_ans_threshold(exact, no_answer_probs,
        qas_id_to_has_answer, no_answer_probability_threshold)
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs,
        qas_id_to_has_answer, no_answer_probability_threshold)
    evaluation = make_eval_dict(exact_threshold, f1_threshold)
    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold,
            qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, 'HasAns')
    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold,
            qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, 'NoAns')
    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs,
            qas_id_to_has_answer)
    return evaluation
