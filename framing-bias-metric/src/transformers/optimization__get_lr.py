@staticmethod
def _get_lr(param_group, param_state):
    rel_step_sz = param_group['lr']
    if param_group['relative_step']:
        min_step = 1e-06 * param_state['step'] if param_group['warmup_init'
            ] else 0.01
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
    param_scale = 1.0
    if param_group['scale_parameter']:
        param_scale = max(param_group['eps'][1], param_state['RMS'])
    return param_scale * rel_step_sz
