def on_epoch_end(self, epoch, logs={}):
    msg = '[Epoch: %i] %s' % (epoch, ', '.join('%s: %f' % (k, v) for k, v in
        sorted(logs.items())))
    self.print_fcn(msg)
