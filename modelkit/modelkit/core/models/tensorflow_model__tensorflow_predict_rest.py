def _tensorflow_predict_rest(self, vects: Dict[str, 'np.ndarray']) ->Dict[
    str, 'np.ndarray']:
    if not self.requests_session:
        self.requests_session = requests.Session()
    response = self.requests_session.post(
        f'http://{self.service_settings.tf_serving.host}:{self.service_settings.tf_serving.port}/v1/models/{self.tf_model_name}:predict'
        , data=json.dumps({'inputs': vects}, default=safe_np_dump))
    if response.status_code != 200:
        raise TFServingError(
            f'TF Serving error [{response.reason}]: {response.text}')
    response_json = response.json()
    outputs = response_json['outputs']
    if not isinstance(outputs, dict):
        return {name: np.array(outputs, dtype=self.output_dtypes[name]) for
            name in self.output_tensor_mapping}
    return {name: np.array(outputs[name], dtype=self.output_dtypes[name]) for
        name in self.output_tensor_mapping}
