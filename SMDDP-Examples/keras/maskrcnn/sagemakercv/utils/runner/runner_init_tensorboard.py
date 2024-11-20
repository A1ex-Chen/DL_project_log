@master_only
def init_tensorboard(self):
    self.tensorboard_dir = os.path.join(self.work_dir, 'tensorboard')
    os.makedirs(self.tensorboard_dir, exist_ok=True)
