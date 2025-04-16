def training_step(self, batch, batch_nb):
    mode = 'train'
    x, y = batch
    y_hat = self(x)
    loss = F.binary_cross_entropy(y_hat, y)
    accuracy = self.compute_accuracy(y_hat, y)
    self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
    self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
    return {f'loss': loss, f'{mode}_accuracy': accuracy, 'log': {
        f'{mode}_loss': loss}}
