def test3d_mask():
    query_pts = np.array([[787014.438, -340616.906, 6313018.0], [751763.125,
        -59925.969, 6326205.5], [769957.188, -202418.125, 6321069.5]])
    kdtree = KDTree(data_pts_real)
    query_mask = np.zeros(data_pts_real.shape[0])
    query_mask[6:10] = True
    dist, idx = kdtree.query(query_pts, sqr_dists=True, mask=query_mask)
    epsilon = 1e-05
    assert idx[0] == 5
    assert idx[1] == 93
    assert idx[2] == 45
    assert abs(dist[0] - 66759196.1053) < epsilon * dist[0]
    assert abs(dist[1] - 3.0) < epsilon * dist[1]
    assert abs(dist[2] - 20001.0) < epsilon * dist[2]
