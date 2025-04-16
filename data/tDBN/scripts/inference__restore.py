def _restore(self, ckpt_path):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.suffix == '.tckpt'
    torchplus.train.restore(str(ckpt_path), self.net)
