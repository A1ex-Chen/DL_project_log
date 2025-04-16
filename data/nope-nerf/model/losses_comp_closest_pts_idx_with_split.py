def comp_closest_pts_idx_with_split(self, pts_src, pts_des):
    """
        :param pts_src:     (3, S)
        :param pts_des:     (3, D)
        :param num_split:
        :return:
        """
    pts_src_list = torch.split(pts_src, 500000, dim=1)
    idx_list = []
    for pts_src_sec in pts_src_list:
        diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]
        dist = torch.linalg.norm(diff, dim=0)
        closest_idx = torch.argmin(dist, dim=1)
        idx_list.append(closest_idx)
    closest_idx = torch.cat(idx_list)
    return closest_idx
