@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)
