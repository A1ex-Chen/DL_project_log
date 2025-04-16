def flops(self):
    f = 0
    if self.op_ in ['__abs__', '__neg__', '__add__', '__sub__', '__mul__',
        '__radd__', '__rmul__', '__iadd__', '__isub__', '__imul__',
        '__itruediv__', 'abs', 'abs_', 'neg', 'neg_', 'add', 'add_', 'div',
        'div_', 'mul', 'mul_', 'sub', 'sub_', 'exp', 'exp_', 'sign',
        'sign_', 'trunc', 'trunc_', 'sin', 'sin_', 'cos', 'cos_', 'sinh',
        'sinh_', 'cosh', 'cosh_', 'sqrt', 'sqrt_', 'rsqrt', 'rsqrt_',
        '__lt__', '__gt__', '__ge__', '__le__', '__eq__', '__ne__', 'lt',
        'lt_', 'gt', 'gt_', 'ge', 'ge_', 'le', 'le_', 'eq', 'eq_', 'ne',
        'ne_', 'ceil', 'ceil_', 'clamp', 'clamp_', 'floor', 'floor_',
        'round', 'sign', 'sign_', 'trunc', 'trunc_']:
        f = self.elems() / 2
    elif self.op_ in ['fmod', 'fmod_']:
        f = self.elems()
    elif self.op_ in ['tanh', 'tanh_', 'sigmoid', 'sigmoid_', 'log', 'log_',
        'log2', 'log2_', 'log10', 'log10_']:
        f = self.elems() * 2
    elif self.op_ in ['asin', 'asin_', 'acos', 'acos_', 'atan', 'atan_']:
        f = self.elems() * 10
    return f
