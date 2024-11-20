def postprocess_qa_predictions(examples, features, predictions: Tuple[np.
    ndarray, np.ndarray], version_2_with_negative: bool=False, n_best_size:
    int=20, max_answer_length: int=30, null_score_diff_threshold: float=0.0,
    output_dir: Optional[str]=None, prefix: Optional[str]=None, log_level:
    Optional[int]=logging.WARNING):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

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
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError(
            '`predictions` should be a tuple with two elements (start_logits, end_logits).'
            )
    all_start_logits, all_end_logits = predictions
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
    all_predictions_with_offsets = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()
    logger.setLevel(log_level)
    logger.info(
        f'Post-processing {len(examples)} example predictions split into {len(features)} features.'
        )
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]['offset_mapping']
            token_is_max_context = features[feature_index].get(
                'token_is_max_context', None)
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction['score'
                ] > feature_null_score:
                min_null_prediction = {'offsets': (0, 0), 'score':
                    feature_null_score, 'start_logit': start_logits[0],
                    'end_logit': end_logits[0]}
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1
                ].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1
                ].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
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
                        'score': start_logits[start_index] + end_logits[
                        end_index], 'start_logit': start_logits[start_index
                        ], 'end_logit': end_logits[end_index]})
        if version_2_with_negative and min_null_prediction is not None:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction['score']
        predictions = sorted(prelim_predictions, key=lambda x: x['score'],
            reverse=True)[:n_best_size]
        if (version_2_with_negative and min_null_prediction is not None and
            not any(p['offsets'] == (0, 0) for p in predictions)):
            predictions.append(min_null_prediction)
        context = example['context']
        for pred in predictions:
            offsets = pred['offsets']
            pred['text'] = context[offsets[0]:offsets[1]]
        if len(predictions) == 0 or len(predictions) == 1 and predictions[0][
            'text'] == '':
            predictions.insert(0, {'text': 'empty', 'start_logit': 0.0,
                'end_logit': 0.0, 'score': 0.0, 'offsets': (0, 0)})
        scores = np.array([pred.pop('score') for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred['probability'] = prob
        if not version_2_with_negative:
            all_predictions[example['id']] = predictions[0]['text']
            all_predictions_with_offsets[example['id']] = predictions[0]
        else:
            i = 0
            while predictions[i]['text'] == '':
                i += 1
            best_non_null_pred = predictions[i]
            score_diff = null_score - best_non_null_pred['start_logit'
                ] - best_non_null_pred['end_logit']
            scores_diff_json[example['id']] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example['id']] = ''
            else:
                all_predictions[example['id']] = best_non_null_pred['text']
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
    return all_predictions, all_predictions_with_offsets
