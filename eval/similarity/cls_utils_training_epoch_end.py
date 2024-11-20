def training_epoch_end(self, outputs):
    mode = 'train'
    loss_mean = sum([o[f'loss'] for o in outputs]) / len(outputs)
    accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs
        )
    self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
    print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
    self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True,
        on_step=False)
    print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')
