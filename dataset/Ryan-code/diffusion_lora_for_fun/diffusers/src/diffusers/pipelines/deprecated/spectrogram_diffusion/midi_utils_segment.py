def segment(a, n):
    return [a[i:i + n] for i in range(0, len(a), n)]
