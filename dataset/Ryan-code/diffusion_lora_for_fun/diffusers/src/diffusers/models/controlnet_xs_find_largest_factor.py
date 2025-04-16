def find_largest_factor(number, max_factor):
    factor = max_factor
    if factor >= number:
        return number
    while factor != 0:
        residual = number % factor
        if residual == 0:
            return factor
        factor -= 1
