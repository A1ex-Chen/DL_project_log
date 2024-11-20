def log(self, step, data):
    for k, v in data.items():
        self.log_value(step, k, v.item() if type(v) is torch.Tensor else v)
