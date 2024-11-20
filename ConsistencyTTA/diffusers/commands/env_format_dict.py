@staticmethod
def format_dict(d):
    return '\n'.join([f'- {prop}: {val}' for prop, val in d.items()]) + '\n'
