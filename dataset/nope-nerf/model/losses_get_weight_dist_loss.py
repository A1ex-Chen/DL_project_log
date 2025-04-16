def get_weight_dist_loss(self, t_list):
    dist = t_list - t_list.roll(shifts=1, dims=0)
    dist = dist[1:]
    dist = dist.norm(dim=1)
    dist_diff = dist - dist.roll(shifts=1)
    dist_diff = dist_diff[1:]
    loss_dist_1st = dist.mean()
    loss_dist_2nd = dist_diff.pow(2.0).mean()
    return loss_dist_1st, loss_dist_2nd
