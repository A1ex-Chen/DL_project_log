def test_epoch_end(self, outputs):
    mode = 'test'
    loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
    accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs
        )
    self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
    print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
    self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True,
        on_step=False)
    print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')
    with open('../results/predictions_{}.txt'.format(datetime.now().
        strftime('%Y-%m-%d_%H:%M:%S')), 'w') as output_file:
        [output_file.write('Idx {}: GT: {} -- Pred: {} -- Conf: {:.4f}\n'.
            format(idx, y, (y_hat >= 0.5).float(), y_hat)) for idx, (y,
            y_hat) in enumerate(zip(self.test_y, self.test_y_hat))]
