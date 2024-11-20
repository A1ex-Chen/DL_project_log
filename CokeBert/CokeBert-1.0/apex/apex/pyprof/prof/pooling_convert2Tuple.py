def convert2Tuple(arg):
    assert arg['type'] in ['int', 'tuple']
    if arg['type'] == 'int':
        return arg['value'], arg['value']
    else:
        return arg['value']
