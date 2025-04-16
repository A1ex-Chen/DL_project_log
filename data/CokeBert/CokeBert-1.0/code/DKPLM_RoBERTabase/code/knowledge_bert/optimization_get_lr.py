def get_lr(self):
    lr = []
    for group in self.param_groups:
        for p in group['params']:
            state = self.state[p]
            if len(state) == 0:
                return [0]
            if group['t_total'] != -1:
                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(state['step'] /
                    group['t_total'], group['warmup'])
            else:
                lr_scheduled = group['lr']
            lr.append(lr_scheduled)
    return lr
