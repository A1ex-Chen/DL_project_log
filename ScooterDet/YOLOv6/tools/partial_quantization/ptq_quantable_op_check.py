def quantable_op_check(k, quantable_ops):
    if quantable_ops is None:
        return True
    if k in quantable_ops:
        return True
    else:
        return False
