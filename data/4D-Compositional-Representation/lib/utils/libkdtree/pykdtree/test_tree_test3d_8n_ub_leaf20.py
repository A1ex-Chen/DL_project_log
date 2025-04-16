def test3d_8n_ub_leaf20():
    query_pts = np.array([[787014.438, -340616.906, 6313018.0], [751763.125,
        -59925.969, 6326205.5], [769957.188, -202418.125, 6321069.5]])
    kdtree = KDTree(data_pts_real, leafsize=20)
    dist, idx = kdtree.query(query_pts, k=8, distance_upper_bound=10000.0,
        sqr_dists=False)
    exp_dist = np.array([[0.0, 4052.50235, 4073.89794, 8082.01128, 
        8170.63009, np.Inf, np.Inf, np.Inf], [1.73205081, 2702.16896, 
        2714.31274, 5395.37066, 5437.9321, 8078.55631, 8171.1997, np.Inf],
        [141.424892, 3255.00021, 3442.84958, 6580.19346, 6810.38455, 
        9891.40135, np.Inf, np.Inf]])
    n = 100
    exp_idx = np.array([[7, 8, 6, 9, 5, n, n, n], [93, 94, 92, 95, 91, 96, 
        90, n], [45, 46, 44, 47, 43, 48, n, n]])
    assert np.array_equal(idx, exp_idx)
    assert np.allclose(dist, exp_dist)
