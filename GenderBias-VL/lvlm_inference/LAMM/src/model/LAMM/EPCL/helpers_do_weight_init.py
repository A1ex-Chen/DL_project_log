def do_weight_init(self, weight_init_name):
    func = WEIGHT_INIT_DICT[weight_init_name]
    for _, param in self.named_parameters():
        if param.dim() > 1:
            func(param)
