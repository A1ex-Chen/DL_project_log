def compute_loss(self, model, inputs):
    labels = inputs.pop('labels')
    loss, _ = self._compute_loss(model, inputs, labels)
    return loss
