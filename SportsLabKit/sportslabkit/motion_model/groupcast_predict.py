def predict(self, observations: dict[str, Any], states: dict[str, float |
    np.ndarray[Any, Any]]=...) ->tuple[np.ndarray[Any, Any], dict[str, 
    float | np.ndarray[Any, Any]]]:
    x = torch.tensor(observations.get('pt', None))
    y = self.model(x)
    return y, states
