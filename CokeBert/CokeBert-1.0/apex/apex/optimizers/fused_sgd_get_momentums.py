def get_momentums(self, params):
    momentums = []
    first_run = True
    for p in params:
        param_state = self.state[p]
        if 'momentum_buffer' not in param_state:
            first_run = True
            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            momentums.append(buf)
        else:
            first_run = False
            momentums.append(param_state['momentum_buffer'])
    return momentums, first_run
