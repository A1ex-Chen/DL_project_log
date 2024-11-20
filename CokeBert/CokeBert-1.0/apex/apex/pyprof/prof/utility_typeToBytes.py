@staticmethod
def typeToBytes(t):
    if t in ['uint8', 'int8', 'byte', 'char']:
        return 1
    elif t in ['float16', 'half', 'int16', 'short']:
        return 2
    elif t in ['float32', 'float', 'int32', 'int']:
        return 4
    elif t in ['int64', 'long', 'float64', 'double']:
        return 8
    assert False
