def optimizer_wgc_step(self, closure=None):
    if per_param_clipper is not None:
        for group in self.param_groups:
            for p in group['params']:
                per_param_clipper(p)
    else:
        all_params = itertools.chain(*[g['params'] for g in self.param_groups])
        global_clipper(all_params)
    super(type(self), self).step(closure)
