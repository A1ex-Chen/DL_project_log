def build_faster_rcnn_classification_loss(loss_config):
    """Builds a classification loss for Faster RCNN based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    if not isinstance(loss_config, losses_pb2.ClassificationLoss):
        raise ValueError(
            'loss_config not of type losses_pb2.ClassificationLoss.')
    loss_type = loss_config.WhichOneof('classification_loss')
    if loss_type == 'weighted_sigmoid':
        return losses.WeightedSigmoidClassificationLoss()
    if loss_type == 'weighted_softmax':
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.
            logit_scale)
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.
        logit_scale)
