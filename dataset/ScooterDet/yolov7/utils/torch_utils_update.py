def update(self, model):
    with torch.no_grad():
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.module.state_dict() if is_parallel(model
            ) else model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach()
