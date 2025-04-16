def _generate_optimizer_class_with_gradient_clipping(optimizer: Type[torch.
    optim.Optimizer], *, per_param_clipper: Optional[_GradientClipper]=None,
    global_clipper: Optional[_GradientClipper]=None) ->Type[torch.optim.
    Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert per_param_clipper is None or global_clipper is None, 'Not allowed to use both per-parameter clipping and global clipping'

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group['params']:
                    per_param_clipper(p)
        else:
            all_params = itertools.chain(*[g['params'] for g in self.
                param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)
    OptimizerWithGradientClip = type(optimizer.__name__ +
        'WithGradientClip', (optimizer,), {'step': optimizer_wgc_step})
    return OptimizerWithGradientClip
