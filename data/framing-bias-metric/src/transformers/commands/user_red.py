@classmethod
def red(cls, s):
    return '{}{}{}'.format(cls._bold + cls._red, s, cls._reset)
