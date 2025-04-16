def __repr__(self):
    s = '{name}(groups={groups})'
    return s.format(name=self.__class__.__name__, groups=self.groups)
