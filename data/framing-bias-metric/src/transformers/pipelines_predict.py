def predict(self, X):
    """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
    return self(X=X)
