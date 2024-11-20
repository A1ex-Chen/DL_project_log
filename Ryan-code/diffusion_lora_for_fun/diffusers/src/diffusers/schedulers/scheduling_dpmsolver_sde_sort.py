@staticmethod
def sort(a, b):
    return (a, b, 1) if a < b else (b, a, -1)
