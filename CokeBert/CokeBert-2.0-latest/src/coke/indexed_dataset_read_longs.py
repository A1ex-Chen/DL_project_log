def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a
