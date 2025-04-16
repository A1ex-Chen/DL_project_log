def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:
        return '%ds' % s
    if s < 60 * 60:
        return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60:
        return '%dh %02dm' % (s // (60 * 60), s // 60 % 60)
    if s < 100 * 24 * 60 * 60:
        return '%dd %02dh' % (s // (24 * 60 * 60), s // (60 * 60) % 24)
    return '>100d'
