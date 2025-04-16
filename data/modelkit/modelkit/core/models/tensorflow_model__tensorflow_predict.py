def _tensorflow_predict(self, vects: Dict[str, 'np.ndarray'], grpc_dtype=None
    ) ->Dict[str, 'np.ndarray']:
    """
        a predict_multiple dispatching tf serving requests with the correct mode
        It takes a dictionary of numpy arrays of shape (Nitems, ?) and returns a
         dictionary for the same shape, indexed by self.output_keys
        """
    if self.service_settings.tf_serving.enable:
        if self.service_settings.tf_serving.mode == 'grpc':
            return self._tensorflow_predict_grpc(vects, dtype=grpc_dtype)
        if self.service_settings.tf_serving.mode == 'rest':
            return self._tensorflow_predict_rest(vects)
    return self._tensorflow_predict_local(vects)
