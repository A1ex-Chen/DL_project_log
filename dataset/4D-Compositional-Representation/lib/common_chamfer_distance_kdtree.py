def chamfer_distance_kdtree(points1, points2, give_id=False):
    """ KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    """
    batch_size = points1.size(0)
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)
    chamfer = chamfer1 + chamfer2
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21
    return chamfer
