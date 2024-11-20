def update(self, vars, count=1):
    assert isinstance(vars, dict)
    for key, var in vars.items():
        if key not in self.val_history:
            self.val_history[key] = []
            self.n_history[key] = []
        if tf.is_tensor(var):
            var = var.numpy()
        self.val_history[key].append(var)
        self.n_history[key].append(count)
