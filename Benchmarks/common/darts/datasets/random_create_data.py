def create_data(self, x_dim, num_samples):
    data = [np.random.randn(x_dim).astype('f') for _ in range(num_samples)]
    return np.stack(data)
