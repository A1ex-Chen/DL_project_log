@staticmethod
def get_graph_feature(coor_q, x_q, coor_k, x_k):
    k = 4
    batch_size = x_k.size(0)
    num_points_k = x_k.size(2)
    num_points_q = x_q.size(2)
    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)
        assert idx.shape[1] == k
        idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1
            ) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    num_dims = x_k.size(1)
    x_k = x_k.transpose(2, 1).contiguous()
    feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
    feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0,
        3, 2, 1).contiguous()
    x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
    feature = torch.cat((feature - x_q, x_q), dim=1)
    return feature
