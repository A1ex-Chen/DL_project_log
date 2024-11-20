def update(self, model):
    with torch.no_grad():
        self.updates += 1
        decay = self.decay(self.updates)
        state_dict = model.module.state_dict() if is_parallel(model
            ) else model.state_dict()
        for k, item in self.ema.state_dict().items():
            if item.dtype.is_floating_point:
                item *= decay
                item += (1 - decay) * state_dict[k].detach()
