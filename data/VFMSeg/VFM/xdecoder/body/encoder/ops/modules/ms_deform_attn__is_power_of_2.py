def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.
            format(n, type(n)))
    return n & n - 1 == 0 and n != 0
