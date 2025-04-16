def log_grads(self, step, model):
    if self.enabled:
        norms = [p.grad.norm().item() for p in model.parameters() if p.grad
             is not None]
        for stat in ('max', 'min', 'mean'):
            self.log_value(step, f'grad_{stat}', getattr(np, stat)(norms),
                stat=stat)
