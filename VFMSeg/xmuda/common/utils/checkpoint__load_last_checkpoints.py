def _load_last_checkpoints(self, path):
    last_checkpoints = []
    with open(path, 'r') as f:
        for p in f.readlines():
            if not os.path.isabs(p):
                p = os.path.join(self.save_dir, p)
            last_checkpoints.append(p)
    return last_checkpoints
