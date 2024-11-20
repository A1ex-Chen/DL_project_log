def describe(self) ->str:
    text = '\nConfig:\n'
    text += '\n'.join([f'\t{k} = {v}' for k, v in asdict(self).items()]) + '\n'
    return text
