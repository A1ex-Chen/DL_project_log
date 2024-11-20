def test1d():
    data_pts = np.arange(1000)
    kdtree = KDTree(data_pts, leafsize=15)
    query_pts = np.arange(400, 300, -10)
    dist, idx = kdtree.query(query_pts)
    assert idx[0] == 400
    assert dist[0] == 0
    assert idx[1] == 390
