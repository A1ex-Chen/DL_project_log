@staticmethod
def typeToString(t):
    if t in ['uint8', 'byte', 'char']:
        return 'uint8'
    elif t in ['int8']:
        return 'int8'
    elif t in ['int16', 'short']:
        return 'int16'
    elif t in ['float16', 'half']:
        return 'fp16'
    elif t in ['float32', 'float']:
        return 'fp32'
    elif t in ['int32', 'int']:
        return 'int32'
    elif t in ['int64', 'long']:
        return 'int64'
    elif t in ['float64', 'double']:
        return 'fp64'
    assert False
