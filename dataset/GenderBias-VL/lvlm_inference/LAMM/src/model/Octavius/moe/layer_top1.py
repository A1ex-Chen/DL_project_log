@staticmethod
def top1(tensor):
    values, index = tensor.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index
