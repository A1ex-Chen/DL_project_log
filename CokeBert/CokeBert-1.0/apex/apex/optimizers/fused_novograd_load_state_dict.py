def load_state_dict(self, state_dict):
    super(FusedNovoGrad, self).load_state_dict(state_dict)
    for group in self.param_groups:
        if len(group['params']) > 0:
            group['exp_avg_sq'][0] = group['exp_avg_sq'][0].to(group[
                'params'][0].device)
            group['exp_avg_sq'][1] = group['exp_avg_sq'][1].to(group[
                'params'][0].device)
