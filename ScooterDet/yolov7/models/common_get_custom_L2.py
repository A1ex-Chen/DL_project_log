def get_custom_L2(self):
    K3 = self.rbr_dense.weight_gen()
    K1 = self.rbr_1x1.conv.weight
    t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.
        rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
    t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.
        rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
    l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
    eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
    l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
    return l2_loss_eq_kernel + l2_loss_circle
