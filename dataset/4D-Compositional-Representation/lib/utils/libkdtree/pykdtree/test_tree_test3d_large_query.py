def test3d_large_query():
    query_pts = np.array([[787014.438, -340616.906, 6313018.0], [751763.125,
        -59925.969, 6326205.5], [769957.188, -202418.125, 6321069.5]])
    n = 20000
    query_pts = np.repeat(query_pts, n, axis=0)
    kdtree = KDTree(data_pts_real)
    dist, idx = kdtree.query(query_pts, sqr_dists=True)
    epsilon = 1e-05
    assert np.all(idx[:n] == 7)
    assert np.all(idx[n:2 * n] == 93)
    assert np.all(idx[2 * n:] == 45)
    assert np.all(dist[:n] == 0)
    assert np.all(abs(dist[n:2 * n] - 3.0) < epsilon * dist[n:2 * n])
    assert np.all(abs(dist[2 * n:] - 20001.0) < epsilon * dist[2 * n:])
