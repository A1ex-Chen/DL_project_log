@staticmethod
def ops(input, weight):
    return (input * weight * weight).sum()
