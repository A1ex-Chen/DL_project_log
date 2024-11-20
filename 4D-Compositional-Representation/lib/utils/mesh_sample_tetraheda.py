def sample_tetraheda(tetraheda_points, size):
    N_tetraheda = tetraheda_points.shape[0]
    volume = np.abs(get_tetrahedon_volume(tetraheda_points))
    probs = volume / volume.sum()
    tetraheda_rnd = np.random.choice(range(N_tetraheda), p=probs, size=size)
    tetraheda_rnd_points = tetraheda_points[tetraheda_rnd]
    weights_rnd = np.random.dirichlet([1, 1, 1, 1], size=size)
    weights_rnd = weights_rnd.reshape(size, 4, 1)
    points_rnd = (weights_rnd * tetraheda_rnd_points).sum(axis=1)
    return points_rnd
