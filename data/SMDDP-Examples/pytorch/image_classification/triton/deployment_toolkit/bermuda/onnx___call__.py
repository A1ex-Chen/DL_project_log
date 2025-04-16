def __call__(self, x: Dict[str, object]):
    feed_dict = {k: x[k] for k in self._input_names}
    y_pred = self._session.run(self._output_names, feed_dict)
    y_pred = dict(zip(self._output_names, y_pred))
    return y_pred
