def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    if not isinstance(loss_config, losses_pb2.LocalizationLoss):
        raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.'
            )
    loss_type = loss_config.WhichOneof('localization_loss')
    if loss_type == 'weighted_l2':
        config = loss_config.weighted_l2
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)
    if loss_type == 'weighted_smooth_l1':
        config = loss_config.weighted_smooth_l1
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma,
            code_weight)
    raise ValueError('Empty loss config.')
