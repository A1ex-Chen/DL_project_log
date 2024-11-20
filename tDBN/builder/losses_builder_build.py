def build(loss_config):
    """Build losses based on the config.

  Builds classification, localization losses and optionally a hard example miner
  based on the config.

  Args:
    loss_config: A losses_pb2.Loss object.

  Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.

  Raises:
    ValueError: If hard_example_miner is used with sigmoid_focal_loss.
  """
    classification_loss = _build_classification_loss(loss_config.
        classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight
    hard_example_miner = None
    if loss_config.HasField('hard_example_miner'):
        raise ValueError("Pytorch don't support HardExampleMiner")
    return (classification_loss, localization_loss, classification_weight,
        localization_weight, hard_example_miner)
