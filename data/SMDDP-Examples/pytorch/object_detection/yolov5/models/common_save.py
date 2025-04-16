def save(self, labels=True, save_dir='runs/detect/exp'):
    save_dir = increment_path(save_dir, exist_ok=save_dir !=
        'runs/detect/exp', mkdir=True)
    self.display(save=True, labels=labels, save_dir=save_dir)
