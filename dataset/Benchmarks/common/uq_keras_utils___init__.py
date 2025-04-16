def __init__(self, x, y, a_max=0.99):
    """Initializer of the Contamination_Callback.
        Parameters
        ----------
        x : ndarray
            Array of samples (= input features) in training set.
        y : ndarray
            Array of sample outputs in training set.
        a_max : float
            Maximum value of a variable to allow
        """
    super(Contamination_Callback, self).__init__()
    if y.ndim > 1:
        if y.shape[1] > 1:
            raise Exception(
                'ERROR ! Contamination model can be applied to one-output regression, but provided training data has: '
                 + str(y.ndim) + 'outpus... Exiting')
    self.x = x
    self.y = y
    self.a_max = a_max
    self.sigmaSQ = K.variable(value=0.01)
    self.gammaSQ = K.variable(value=0.01)
    if isinstance(x, list):
        self.T = np.zeros((x[0].shape[0], 2))
    else:
        self.T = np.zeros((self.x.shape[0], 2))
    self.T[:, 0] = np.random.uniform(size=self.T.shape[0])
    self.T[:, 1] = 1.0 - self.T[:, 0]
    self.T_k = K.variable(value=self.T)
    self.a = K.variable(value=np.mean(self.T[:, 0]))
    self.avalues = []
    self.sigmaSQvalues = []
    self.gammaSQvalues = []
