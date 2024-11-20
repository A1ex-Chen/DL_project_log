def __init__(self, func, y0, **kwargs):
    super(AdamsBashforth, self).__init__(func, y0, implicit=False, **kwargs)
