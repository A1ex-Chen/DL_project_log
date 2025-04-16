def test1d_mask():
    data_pts = np.arange(1000)
    np.random.shuffle(data_pts)
    bad_idx = np.nonzero(data_pts == 400)
    nearest_idx_1 = np.nonzero(data_pts == 399)
    nearest_idx_2 = np.nonzero(data_pts == 390)
    kdtree = KDTree(data_pts, leafsize=15)
    query_pts = np.arange(399.9, 299.9, -10)
    query_mask = np.zeros(data_pts.shape[0]).astype(bool)
    query_mask[bad_idx] = True
    dist, idx = kdtree.query(query_pts, mask=query_mask)
    assert idx[0] == nearest_idx_1
    assert np.isclose(dist[0], 0.9)
    assert idx[1] == nearest_idx_2
    assert np.isclose(dist[1], 0.1)
