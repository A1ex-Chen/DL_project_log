def __init__(self, *args, **kwargs):
    self._root_name = kwargs.pop('root_name') + '.'
    self._abbrev_name = kwargs.pop('abbrev_name', '')
    if len(self._abbrev_name):
        self._abbrev_name = self._abbrev_name + '.'
    super(_ColorfulFormatter, self).__init__(*args, **kwargs)
