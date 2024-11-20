def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
    save_dir = increment_path(save_dir, exist_ok, mkdir=True)
    self._run(save=True, labels=labels, save_dir=save_dir)
