@torch.jit.ignore()
def load_pretrained(self, checkpoint_path, prefix=''):
    _load_weights(self, checkpoint_path, prefix)
