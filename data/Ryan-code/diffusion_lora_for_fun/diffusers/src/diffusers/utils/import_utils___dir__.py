def __dir__(self):
    result = super().__dir__()
    for attr in self.__all__:
        if attr not in result:
            result.append(attr)
    return result
