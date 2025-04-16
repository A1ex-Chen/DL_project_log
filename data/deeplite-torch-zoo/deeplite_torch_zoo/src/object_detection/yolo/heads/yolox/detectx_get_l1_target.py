@staticmethod
def get_l1_target(l1_target, gt, stride, xy_shifts, eps=1e-08):
    l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
    l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
    return l1_target
