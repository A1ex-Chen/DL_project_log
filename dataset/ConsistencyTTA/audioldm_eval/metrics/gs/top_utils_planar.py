def planar(N=5000, zdim=32, dim=784):
    A = np.random.rand(N, zdim)
    z = np.random.rand(zdim, dim)
    return np.dot(A, z)
