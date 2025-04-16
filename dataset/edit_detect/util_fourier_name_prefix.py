@property
def name_prefix(self):
    key: str = 'MDID_'
    for var in self.STATIC_VARS:
        if var[:len(key)] == key:
            if getattr(self, var) == self.__model__:
                return f'{var[len(key):]}_'
