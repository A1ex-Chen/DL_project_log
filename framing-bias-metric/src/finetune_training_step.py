def training_step(self, batch, batch_idx) ->Dict:
    loss_tensors = self._step(batch)
    logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
    return {'loss': loss_tensors[0], 'log': logs}
