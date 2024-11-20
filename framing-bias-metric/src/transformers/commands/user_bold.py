@classmethod
def bold(cls, s):
    return '{}{}{}'.format(cls._bold, s, cls._reset)
