def rel_error(true, estimate):
    return max_abs((true - estimate) / true)
