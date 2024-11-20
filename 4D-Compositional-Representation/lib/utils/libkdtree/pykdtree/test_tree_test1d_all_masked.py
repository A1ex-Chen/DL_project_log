def test1d_all_masked():
    data_pts = np.arange(1000)
    np.random.shuffle(data_pts)
    kdtree = KDTree(data_pts, leafsize=15)
    query_pts = np.arange(400, 300, -10)
    query_mask = np.ones(data_pts.shape[0]).astype(bool)
    dist, idx = kdtree.query(query_pts, mask=query_mask)
    assert np.all(i >= 1000 for i in idx)
    assert np.all(d >= 1001 for d in dist)
