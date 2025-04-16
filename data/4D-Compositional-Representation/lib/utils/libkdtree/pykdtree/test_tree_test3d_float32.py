def test3d_float32():
    query_pts = np.array([[787014.438, -340616.906, 6313018.0], [751763.125,
        -59925.969, 6326205.5], [769957.188, -202418.125, 6321069.5]],
        dtype=np.float32)
    kdtree = KDTree(data_pts_real.astype(np.float32))
    dist, idx = kdtree.query(query_pts, sqr_dists=True)
    epsilon = 1e-05
    assert idx[0] == 7
    assert idx[1] == 93
    assert idx[2] == 45
    assert dist[0] == 0
    assert abs(dist[1] - 3.0) < epsilon * dist[1]
    assert abs(dist[2] - 20001.0) < epsilon * dist[2]
    assert kdtree.data_pts.dtype == np.float32
