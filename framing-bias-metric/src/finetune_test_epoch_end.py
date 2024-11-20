def test_epoch_end(self, outputs):
    return self.validation_epoch_end(outputs, prefix='test')
