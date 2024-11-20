def choose_bytes_unit(num_bytes):
    b = int(np.rint(num_bytes))
    if b < 100 << 0:
        return 'B', 1 << 0
    if b < 100 << 10:
        return 'kB', 1 << 10
    if b < 100 << 20:
        return 'MB', 1 << 20
    if b < 100 << 30:
        return 'GB', 1 << 30
    return 'TB', 1 << 40
