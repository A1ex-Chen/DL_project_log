@torch.jit.ignore
def to_list_3d(arr) ->List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr
