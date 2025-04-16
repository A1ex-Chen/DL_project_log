def compute_metrics(p: EvalPrediction):
    if 'phrase_sense_disambiguation' not in data_args.dataset_name.lower():
        return metric.compute(predictions=p.predictions, references=p.label_ids
            )
    formatted_predictions = [{'id': pred['id'], 'prediction_text': pred[
        'prediction_text']} for pred in p.predictions]
    em_with_index_count, f1_with_index_count = 0, 0
    em_count, f1_count = 0, 0
    for pred, label in zip(p.predictions, p.label_ids):
        assert pred['id'] == label['id']
        answer_text = label['answers']['text'][0]
        answer_start = label['answers']['answer_start'][0]
        if pred['prediction_text'].lower() == answer_text.lower():
            em_count += 1
            if pred['answer_start'] == answer_start:
                em_with_index_count += 1
            else:
                print({'id': pred['id'], 'pred_text': [pred[
                    'prediction_text'], pred['answer_start']],
                    'answer_text': [answer_text, answer_start]})
        f1_count += compute_f1(answer_text, pred['prediction_text'])
        pred_range = range(pred['answer_start'], pred['answer_start'] + len
            (pred['prediction_text']))
        answer_range = range(answer_start, answer_start + len(answer_text))
        if len(set(pred_range).intersection(set(answer_range))) > 0:
            f1_with_index_count += compute_f1(answer_text, pred[
                'prediction_text'])
    em_score = round(em_count / len(p.label_ids), 4) * 100
    f1_score = round(f1_count / len(p.label_ids), 4) * 100
    em_with_index_score = round(em_with_index_count / len(p.label_ids), 4
        ) * 100
    f1_with_index_score = round(f1_with_index_count / len(p.label_ids), 4
        ) * 100
    print('Computed results: {}'.format({'exact_match': em_score,
        'exact_match_with_index': em_with_index_score, 'f1': f1_score,
        'f1_with_index': f1_with_index_score}))
    print('Reference results: {}'.format(metric.compute(predictions=
        formatted_predictions, references=p.label_ids)))
    return {'exact_match': em_with_index_score, 'f1': f1_with_index_score}
