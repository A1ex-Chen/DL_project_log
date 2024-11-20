def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

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
    if loss_type == 'weighted_sigmoid_focal':
        config = loss_config.weighted_sigmoid_focal
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(gamma=config.gamma,
            alpha=alpha)
    if loss_type == 'weighted_softmax_focal':
        config = loss_config.weighted_softmax_focal
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SoftmaxFocalClassificationLoss(gamma=config.gamma,
            alpha=alpha)
    if loss_type == 'weighted_softmax':
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.
            logit_scale)
    if loss_type == 'bootstrapped_sigmoid':
        config = loss_config.bootstrapped_sigmoid
        return losses.BootstrappedSigmoidClassificationLoss(alpha=config.
            alpha, bootstrap_type='hard' if config.hard_bootstrap else 'soft')
    raise ValueError('Empty loss config.')
