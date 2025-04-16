@torch.no_grad()
def step(self, parameters):
    parameters = list(parameters)
    self.optimization_step += 1
    self.decay = self.get_decay(self.optimization_step)
    for s_param, param in zip(self.shadow_params, parameters):
        if param.requires_grad:
            tmp = self.decay * (s_param - param)
            s_param.sub_(tmp)
        else:
            s_param.copy_(param)
    torch.cuda.empty_cache()
