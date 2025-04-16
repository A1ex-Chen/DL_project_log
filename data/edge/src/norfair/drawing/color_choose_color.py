@classmethod
def choose_color(cls, hashable: Hashable) ->ColorType:
    if hashable is None:
        return cls._default_color
    return cls._colors[abs(hash(hashable)) % len(cls._colors)]
