@property
def name_or_path(self) ->str:
    return getattr(self.config, '_name_or_path', None)
