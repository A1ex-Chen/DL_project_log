def test3d_float32_mismatch():
    query_pts = np.array([[787014.438, -340616.906, 6313018.0], [751763.125,
        -59925.969, 6326205.5], [769957.188, -202418.125, 6321069.5]],
        dtype=np.float32)
    kdtree = KDTree(data_pts_real)
    dist, idx = kdtree.query(query_pts, sqr_dists=True)
