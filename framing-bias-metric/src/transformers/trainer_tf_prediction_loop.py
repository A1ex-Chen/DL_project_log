def prediction_loop(self, dataset: tf.data.Dataset, steps: int,
    num_examples: int, description: str, prediction_loss_only: Optional[
    bool]=None) ->PredictionOutput:
    """
        Prediction/evaluation loop, shared by :func:`~transformers.TFTrainer.evaluate` and
        :func:`~transformers.TFTrainer.predict`.

        Works both with or without labels.
        """
    prediction_loss_only = (prediction_loss_only if prediction_loss_only is not
        None else self.args.prediction_loss_only)
    logger.info('***** Running %s *****', description)
    logger.info('  Num examples = %d', num_examples)
    logger.info('  Batch size = %d', self.args.eval_batch_size)
    label_ids: np.ndarray = None
    preds: np.ndarray = None
    self.eval_loss = tf.keras.metrics.Sum()
    if self.args.past_index >= 0:
        self._past = None
    for step, batch in enumerate(dataset):
        logits = self.distributed_prediction_steps(batch)
        _, labels = batch
        if not prediction_loss_only:
            if isinstance(logits, tuple):
                logits = logits[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            if self.args.n_replicas > 1:
                for val in logits.values:
                    if preds is None:
                        preds = val.numpy()
                    else:
                        preds = np.append(preds, val.numpy(), axis=0)
                for val in labels.values:
                    if label_ids is None:
                        label_ids = val.numpy()
                    else:
                        label_ids = np.append(label_ids, val.numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.numpy()
                else:
                    preds = np.append(preds, logits.numpy(), axis=0)
                if label_ids is None:
                    label_ids = labels.numpy()
                else:
                    label_ids = np.append(label_ids, labels.numpy(), axis=0)
            if step == steps:
                break
    if (self.compute_metrics is not None and preds is not None and 
        label_ids is not None):
        metrics = self.compute_metrics(EvalPrediction(predictions=preds,
            label_ids=label_ids))
    else:
        metrics = {}
    metrics['eval_loss'] = self.eval_loss.result().numpy() / steps
    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics[f'eval_{key}'] = metrics.pop(key)
    if self.args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
    return PredictionOutput(predictions=preds, label_ids=label_ids, metrics
        =metrics)
