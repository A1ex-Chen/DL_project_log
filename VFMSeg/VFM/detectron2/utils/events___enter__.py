def __enter__(self):
    _CURRENT_STORAGE_STACK.append(self)
    return self
