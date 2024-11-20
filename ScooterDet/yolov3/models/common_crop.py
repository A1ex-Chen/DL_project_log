def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
    save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
    return self._run(crop=True, save=save, save_dir=save_dir)
