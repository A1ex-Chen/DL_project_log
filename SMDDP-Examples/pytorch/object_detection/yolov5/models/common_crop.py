def crop(self, save=True, save_dir='runs/detect/exp'):
    save_dir = increment_path(save_dir, exist_ok=save_dir !=
        'runs/detect/exp', mkdir=True) if save else None
    return self.display(crop=True, save=save, save_dir=save_dir)
