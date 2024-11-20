def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = deepcopy(mask)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, i - 1, j] = 0
                if j != 0:
                    mask[:, i, j - 1] = 0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        prepared_masks.append(mask)
    return torch.stack(prepared_masks, dim=0)
