@staticmethod
def get_best_param_group_id(optimizer):
    largest_group = max(len(g['params']) for g in optimizer.param_groups)
    if largest_group == 1:
        lr_count = Counter([g['lr'] for g in optimizer.param_groups])
        lr = lr_count.most_common()[0][0]
        for i, g in enumerate(optimizer.param_groups):
            if g['lr'] == lr:
                return i
    else:
        for i, g in enumerate(optimizer.param_groups):
            if len(g['params']) == largest_group:
                return i
