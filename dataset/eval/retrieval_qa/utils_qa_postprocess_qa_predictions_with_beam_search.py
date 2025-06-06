def postprocess_qa_predictions_with_beam_search(examples, features,
    predictions: Tuple[np.ndarray, np.ndarray], version_2_with_negative:
    bool=False, n_best_size: int=20, max_answer_length: int=30, start_n_top:
    int=5, end_n_top: int=5, output_dir: Optional[str]=None, prefix:
    Optional[str]=None, log_level: Optional[int]=logging.WARNING):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 5:
        raise ValueError('`predictions` should be a tuple with five elements.')
    (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index,
        cls_logits) = predictions
    if len(predictions[0]) != len(features):
        raise ValueError(
            f'Got {len(predictions[0])} predictions and {len(features)} features.'
            )
    example_id_to_index = {k: i for i, k in enumerate(examples['id'])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature['example_id']]
            ].append(i)
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict(
        ) if version_2_with_negative else None
    logger.setLevel(log_level)
    logger.info(
        f'Post-processing {len(examples)} example predictions split into {len(features)} features.'
        )
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']
            token_is_max_context = features[feature_index].get(
                'token_is_max_context', None)
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    if start_index >= len(offset_mapping) or end_index >= len(
                        offset_mapping) or offset_mapping[start_index
                        ] is None or len(offset_mapping[start_index]
                        ) < 2 or offset_mapping[end_index] is None or len(
                        offset_mapping[end_index]) < 2:
                        continue
                    if (end_index < start_index or end_index - start_index +
                        1 > max_answer_length):
                        continue
                    if (token_is_max_context is not None and not
                        token_is_max_context.get(str(start_index), False)):
                        continue
                    prelim_predictions.append({'offsets': (offset_mapping[
                        start_index][0], offset_mapping[end_index][1]),
                        'score': start_log_prob[i] + end_log_prob[j_index],
                        'start_log_prob': start_log_prob[i], 'end_log_prob':
                        end_log_prob[j_index]})
        predictions = sorted(prelim_predictions, key=lambda x: x['score'],
            reverse=True)[:n_best_size]
        context = example['context']
        for pred in predictions:
            offsets = pred.pop('offsets')
            pred['text'] = context[offsets[0]:offsets[1]]
        if len(predictions) == 0:
            min_null_score = -2e-06
            predictions.insert(0, {'text': '', 'start_logit': -1e-06,
                'end_logit': -1e-06, 'score': min_null_score})
        scores = np.array([pred.pop('score') for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred['probability'] = prob
        all_predictions[example['id']] = predictions[0]['text']
        if version_2_with_negative:
            scores_diff_json[example['id']] = float(min_null_score)
        all_nbest_json[example['id']] = [{k: (float(v) if isinstance(v, (np
            .float16, np.float32, np.float64)) else v) for k, v in pred.
            items()} for pred in predictions]
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f'{output_dir} is not a directory.')
        prediction_file = os.path.join(output_dir, 'predictions.json' if 
            prefix is None else f'{prefix}_predictions.json')
        nbest_file = os.path.join(output_dir, 'nbest_predictions.json' if 
            prefix is None else f'{prefix}_nbest_predictions.json')
        if version_2_with_negative:
            null_odds_file = os.path.join(output_dir, 'null_odds.json' if 
                prefix is None else f'{prefix}_null_odds.json')
        logger.info(f'Saving predictions to {prediction_file}.')
        with open(prediction_file, 'w') as writer:
            writer.write(json.dumps(all_predictions, indent=4) + '\n')
        logger.info(f'Saving nbest_preds to {nbest_file}.')
        with open(nbest_file, 'w') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
        if version_2_with_negative:
            logger.info(f'Saving null_odds to {null_odds_file}.')
            with open(null_odds_file, 'w') as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + '\n')
    return all_predictions, scores_diff_json
