def _reset_parameters(self, weight_init_name):
    func = WEIGHT_INIT_DICT[weight_init_name]
    for p in self.parameters():
        if p.dim() > 1:
            func(p)
