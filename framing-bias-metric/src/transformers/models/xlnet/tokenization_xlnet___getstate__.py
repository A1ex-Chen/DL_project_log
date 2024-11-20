def __getstate__(self):
    state = self.__dict__.copy()
    state['sp_model'] = None
    return state
