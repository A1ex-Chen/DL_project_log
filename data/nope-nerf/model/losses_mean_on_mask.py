def mean_on_mask(self, diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 0:
        mean_value = diff[mask].sum() / mask.sum()
    else:
        print('============invalid mask==========')
        mean_value = torch.tensor(0).float().cuda()
    return mean_value
