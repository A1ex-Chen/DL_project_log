def format_time(t):
    """Format `t` (in seconds) to (h):mm:ss"""
    t = int(t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f'{h}:{m:02d}:{s:02d}' if h != 0 else f'{m:02d}:{s:02d}'
