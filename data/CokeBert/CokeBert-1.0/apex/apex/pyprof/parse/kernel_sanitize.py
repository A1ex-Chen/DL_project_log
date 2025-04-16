def sanitize(name):
    name = name.replace('torch', '').replace('autograd', '').replace(
        '_backward', '').replace('::', '').replace('jit', '').replace(
        '(anonymous namespace)', '')
    head, sep, tail = name.partition('Backward')
    return head
