def post_processing_function(examples, features, predictions, stage='eval'):
    predictions, predictions_with_indices = postprocess_qa_predictions(examples
        =examples, features=features, predictions=predictions,
        version_2_with_negative=data_args.version_2_with_negative,
        n_best_size=data_args.n_best_size, max_answer_length=data_args.
        max_answer_length, null_score_diff_threshold=data_args.
        null_score_diff_threshold, output_dir=training_args.output_dir,
        log_level=log_level, prefix=stage)
    if data_args.version_2_with_negative:
        formatted_predictions = [{'id': k, 'prediction_text': v,
            'no_answer_probability': 0.0} for k, v in predictions.items()]
    elif 'phrase_sense_disambiguation' not in data_args.dataset_name.lower():
        formatted_predictions = [{'id': k, 'prediction_text': v} for k, v in
            predictions.items()]
    else:
        formatted_predictions = [{'id': k, 'prediction_text': v,
            'answer_start': predictions_with_indices[k]['offsets'][0]} for 
            k, v in predictions.items()]
    references = [{'id': ex['id'], 'answers': ex[answer_column_name]} for
        ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=
        references)
