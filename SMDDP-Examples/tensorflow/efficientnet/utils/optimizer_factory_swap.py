def swap(strategy, a_and_b):
    """Swap `a` and `b` and mirror to all devices."""
    for a, b in a_and_b:
        strategy.extended.update(a, fn_0, args=(b,))
        strategy.extended.update(b, fn_1, args=(a,))
        strategy.extended.update(a, fn_2, args=(b,))
