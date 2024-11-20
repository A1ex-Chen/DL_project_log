@staticmethod
def ops(input, weight0, weight1):
    return (input * weight0.float() * weight1.float()).sum()
