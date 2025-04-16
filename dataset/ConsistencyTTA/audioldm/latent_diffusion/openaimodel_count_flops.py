@staticmethod
def count_flops(model, _x, y):
    return count_flops_attn(model, _x, y)
