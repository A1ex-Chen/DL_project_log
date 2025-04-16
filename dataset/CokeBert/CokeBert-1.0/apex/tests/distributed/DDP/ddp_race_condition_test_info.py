def info(name, param, val):
    expected = val * 4096 * 4096 * (2.0 * i + 1) / 2.0
    actual = param.grad.data.sum().item()
    print(name + ': grad.data_ptr() = {}, expected sum {}, got {}'.format(
        param.grad.data_ptr(), expected, actual))
    return expected == actual
