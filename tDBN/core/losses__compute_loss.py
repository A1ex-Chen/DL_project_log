def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    if self._bootstrap_type == 'soft':
        bootstrap_target_tensor = self._alpha * target_tensor + (1.0 - self
            ._alpha) * torch.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = self._alpha * target_tensor + (1.0 - self
            ._alpha) * (torch.sigmoid(prediction_tensor) > 0.5).float()
    per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(labels=
        bootstrap_target_tensor, logits=prediction_tensor)
    return per_entry_cross_ent * weights.unsqueeze(2)
