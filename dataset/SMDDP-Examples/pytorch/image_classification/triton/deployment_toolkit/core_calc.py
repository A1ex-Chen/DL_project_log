@abc.abstractmethod
def calc(self, *, ids: List[Any], y_pred: Dict[str, np.ndarray], x:
    Optional[Dict[str, np.ndarray]], y_real: Optional[Dict[str, np.ndarray]]
    ) ->Dict[str, float]:
    """
        Calculates error/accuracy metrics
        Args:
            ids: List of ids identifying each sample in the batch
            y_pred: model output as dict where key is output name and value is output value
            x: model input as dict where key is input name and value is input value
            y_real: input ground truth as dict where key is output name and value is output value
        Returns:
            dictionary where key is metric name and value is its value
        """
    pass
