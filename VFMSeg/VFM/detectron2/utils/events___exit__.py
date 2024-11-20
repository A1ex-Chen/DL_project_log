def __exit__(self, exc_type, exc_val, exc_tb):
    assert _CURRENT_STORAGE_STACK[-1] == self
    _CURRENT_STORAGE_STACK.pop()
