def stdout_step_format(step):
    if isinstance(step, str):
        return step
    fields = []
    if len(step) > 0:
        fields.append('epoch {:>4}'.format(step[0]))
    if len(step) > 1:
        fields.append('iter {:>4}'.format(step[1]))
    if len(step) > 2:
        fields[-1] += '/{}'.format(step[2])
    return ' | '.join(fields)
