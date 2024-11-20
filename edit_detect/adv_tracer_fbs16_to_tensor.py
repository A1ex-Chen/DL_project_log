def to_tensor(self):
    res: List = list(self[0:len(self)])
    for i in range(len(res)):
        if isinstance(res[i][0], torch.Tensor):
            res[i] = torch.stack(res[i], dim=0)
        elif isinstance(res[i][0], int) or isinstance(res[i][0], float):
            res[i] = torch.tensor(res[i])
    return res
