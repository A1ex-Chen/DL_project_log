def _aug(self, img, ops, apply_or_not):
    for i, (name, level) in enumerate(ops):
        if not apply_or_not[i]:
            continue
        args = arg_dict[name](level)
        img = func_dict[name](img, *args)
    return torch.from_numpy(img)
