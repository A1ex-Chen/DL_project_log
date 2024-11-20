@torch.jit.ignore()
def load_pretrained(self, checkpoint_path: str, prefix: str='') ->None:
    _load_weights(self, checkpoint_path, prefix)
