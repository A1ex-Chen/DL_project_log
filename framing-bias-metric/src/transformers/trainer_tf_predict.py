def predict(self, test_dataset: tf.data.Dataset) ->PredictionOutput:
    """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on. The dataset should yield tuples of ``(features, labels)`` where
                ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the
                loss is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict,
                such as when using a QuestionAnswering head model with multiple targets, the loss is instead calculated
                by calling ``model(features, **labels)``

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
    test_ds, steps, num_examples = self.get_test_tfdataset(test_dataset)
    return self.prediction_loop(test_ds, steps, num_examples, description=
        'Prediction')
