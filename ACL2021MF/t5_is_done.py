@property
def is_done(self) ->bool:
    return self._done.all()
