def ctaTile(name):
    name = name.split('_')
    name = list(filter(lambda x: 'x' in x, name))
    name = list(filter(lambda x: 'slice' not in x, name))
    assert len(name) == 1
    name = name[0].split('x')
    assert len(name) == 2
    name = list(map(int, name))
    return name[0], name[1]
