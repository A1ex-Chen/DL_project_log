def test_step(self, batch, batch_nb):
    mode = 'test'
    x, y = batch
    y_hat = self(x)
    loss = F.binary_cross_entropy(y_hat, y)
    accuracy = self.compute_accuracy(y_hat, y)
    self.test_y.extend(y)
    self.test_y_hat.extend(y_hat)
    self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
    self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
    return {f'{mode}_loss': loss, f'{mode}_accuracy': accuracy, 'log': {
        f'{mode}_loss': loss}}
