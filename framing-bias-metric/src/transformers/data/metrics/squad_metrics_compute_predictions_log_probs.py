def compute_predictions_log_probs(all_examples, all_features, all_results,
    n_best_size, max_answer_length, output_prediction_file,
    output_nbest_file, output_null_log_odds_file, start_n_top, end_n_top,
    version_2_with_negative, tokenizer, verbose_logging):
    """
    XLNet write prediction logic (more complex than Bert's). Write final predictions to the json file and log-odds of
    null if needed.

    Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', [
        'feature_index', 'start_index', 'end_index', 'start_log_prob',
        'end_log_prob'])
    _NbestPrediction = collections.namedtuple('NbestPrediction', ['text',
        'start_log_prob', 'end_log_prob'])
    logger.info('Writing predictions to: %s', output_prediction_file)
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits
            score_null = min(score_null, cur_null_score)
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]
                    j_index = i * end_n_top + j
                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue
                    if not feature.token_is_max_context.get(start_index, False
                        ):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(_PrelimPrediction(
                        feature_index=feature_index, start_index=
                        start_index, end_index=end_index, start_log_prob=
                        start_log_prob, end_log_prob=end_log_prob))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x.
            start_log_prob + x.end_log_prob, reverse=True)
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:pred.end_index + 1]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:orig_doc_end + 1]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)
            if hasattr(tokenizer, 'do_lower_case'):
                do_lower_case = tokenizer.do_lower_case
            else:
                do_lower_case = tokenizer.do_lowercase_and_remove_accent
            final_text = get_final_text(tok_text, orig_text, do_lower_case,
                verbose_logging)
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
            nbest.append(_NbestPrediction(text=final_text, start_log_prob=
                pred.start_log_prob, end_log_prob=pred.end_log_prob))
        if not nbest:
            nbest.append(_NbestPrediction(text='', start_log_prob=-
                1000000.0, end_log_prob=-1000000.0))
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_log_prob'] = entry.start_log_prob
            output['end_log_prob'] = entry.end_log_prob
            nbest_json.append(output)
        assert len(nbest_json) >= 1, 'No valid predictions'
        assert best_non_null_entry is not None, 'No valid predictions'
        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if version_2_with_negative:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions
