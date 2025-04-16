@abstractmethod
def predict(self, observations: (float | np.ndarray), states: (float | np.
    ndarray | None)) ->tuple[float | np.ndarray | None, float | np.ndarray]:
    """Compute the next internal state and prediction based on the current observation and internal state.

        Args:
            observation (Union[float, np.ndarray]): The current observation.
            states (Union[float, np.ndarray, None]): The current internal state of the motion model.

        Returns:
            Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray]]: The next internal state and the prediction.
        """
    pass
