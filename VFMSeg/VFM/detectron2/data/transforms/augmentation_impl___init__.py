def __init__(self, scale):
    """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
    super().__init__()
    self._init(locals())
    self.eigen_vecs = np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -
        0.0045, -0.814], [-0.5836, -0.6948, 0.4203]])
    self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
