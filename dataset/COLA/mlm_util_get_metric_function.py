def get_metric_function():
    metric = evaluate.load('accuracy')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)
    return compute_metrics
