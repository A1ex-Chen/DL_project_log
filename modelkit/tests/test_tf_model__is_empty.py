def _is_empty(self, item):
    if item['input_1'][0, 0, 0] == -1:
        return True
    return False
