def format_step(step):
    if isinstance(step, str):
        return step
    s = ''
    if len(step) > 0:
        if isinstance(step[0], Number):
            s += 'Epoch: {} '.format(step[0])
        else:
            s += '{} '.format(step[0])
    if len(step) > 1:
        s += 'Iteration: {} '.format(step[1])
    if len(step) > 2:
        s += 'Validation Iteration: {} '.format(step[2])
    if len(step) == 0:
        s = 'Summary:'
    return s
