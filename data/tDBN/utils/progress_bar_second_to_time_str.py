def second_to_time_str(second, omit_hours_if_possible=True):
    second = int(second)
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    if omit_hours_if_possible:
        if h == 0:
            return '{:02d}:{:02d}'.format(m, s)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
