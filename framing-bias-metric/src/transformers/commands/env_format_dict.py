@staticmethod
def format_dict(d):
    return '\n'.join(['- {}: {}'.format(prop, val) for prop, val in d.items()]
        ) + '\n'
