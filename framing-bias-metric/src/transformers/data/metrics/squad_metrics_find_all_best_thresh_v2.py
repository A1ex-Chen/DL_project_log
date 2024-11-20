def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs,
    qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds,
        exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw,
        na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh
    main_eval['has_ans_exact'] = has_ans_exact
    main_eval['has_ans_f1'] = has_ans_f1
