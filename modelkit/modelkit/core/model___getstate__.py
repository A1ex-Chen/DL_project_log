def __getstate__(self):
    state = copy.deepcopy(self.__dict__)
    state['_item_model'] = None
    state['_return_model'] = None
    return state
