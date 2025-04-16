def save(self, save_dir='runs/hub/exp'):
    save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    self.display(save=True, save_dir=save_dir)
