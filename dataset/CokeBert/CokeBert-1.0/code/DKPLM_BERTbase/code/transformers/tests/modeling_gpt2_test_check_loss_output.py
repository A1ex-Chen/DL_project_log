def check_loss_output(self, result):
    self.parent.assertListEqual(list(result['loss'].size()), [])
