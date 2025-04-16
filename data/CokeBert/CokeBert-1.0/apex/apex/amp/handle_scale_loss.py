@contextlib.contextmanager
def scale_loss(self, loss, optimizer):
    yield loss
