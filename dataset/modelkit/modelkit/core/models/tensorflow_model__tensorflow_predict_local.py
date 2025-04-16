def _tensorflow_predict_local(self, vects: Dict[str, 'np.ndarray']) ->Dict[
    str, 'np.ndarray']:
    results = self.tf_model_signature(**{key: tf.convert_to_tensor(value) for
        key, value in vects.items()})
    return {name: np.array(results[name], dtype=self.output_dtypes.get(name
        )) for name in self.output_tensor_mapping}
