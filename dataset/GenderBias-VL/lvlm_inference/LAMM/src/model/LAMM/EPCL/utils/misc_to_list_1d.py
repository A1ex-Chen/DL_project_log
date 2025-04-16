@torch.jit.ignore
def to_list_1d(arr) ->List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr
