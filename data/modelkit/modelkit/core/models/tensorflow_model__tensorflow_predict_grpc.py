def _tensorflow_predict_grpc(self, vects: Dict[str, 'np.ndarray'], dtype=None
    ) ->Dict[str, 'np.ndarray']:
    request = PredictRequest()
    request.model_spec.name = self.tf_model_name
    for key, vect in vects.items():
        request.inputs[key].CopyFrom(tf.compat.v1.make_tensor_proto(vect,
            dtype=dtype))
    if not self.grpc_stub:
        self.grpc_stub = connect_tf_serving_grpc(self.tf_model_name, self.
            service_settings.tf_serving.host, self.service_settings.
            tf_serving.port)
    r = self.grpc_stub.Predict(request, 1)
    return {output_key: np.array(r.outputs[output_key].ListFields()[-1][1],
        dtype=self.output_dtypes.get(output_key)).reshape((vect.shape[0],) +
        self.output_shapes[output_key]) for output_key in self.
        output_tensor_mapping}
