def req_loop(self):
    client = InferenceServerClient(self._server_url, verbose=self._verbose)
    self._errors = self._verify_triton_state(client)
    if self._errors:
        return
    LOGGER.debug(
        f'Triton server {self._server_url} and model {self._model_name}:{self._model_version} are up and ready!'
        )
    model_config = client.get_model_config(self._model_name, self.
        _model_version)
    model_metadata = client.get_model_metadata(self._model_name, self.
        _model_version)
    LOGGER.info(f'Model config {model_config}')
    LOGGER.info(f'Model metadata {model_metadata}')
    inputs = {tm.name: tm for tm in model_metadata.inputs}
    outputs = {tm.name: tm for tm in model_metadata.outputs}
    output_names = list(outputs)
    outputs_req = [InferRequestedOutput(name) for name in outputs]
    self._num_waiting_for = 0
    for ids, x, y_real in self._dataloader:
        infer_inputs = []
        for name in inputs:
            data = x[name]
            infer_input = InferInput(name, data.shape, inputs[name].datatype)
            target_np_dtype = client_utils.triton_to_np_dtype(inputs[name].
                datatype)
            data = data.astype(target_np_dtype)
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)
        with self._sync:

            def _check_can_send():
                return self._num_waiting_for < self._max_unresp_reqs
            can_send = self._sync.wait_for(_check_can_send, timeout=self.
                _response_wait_t)
            if not can_send:
                error_msg = (
                    f'Runner could not send new requests for {self._response_wait_t}s'
                    )
                self._errors.append(error_msg)
                break
            callback = functools.partial(AsyncGRPCTritonRunner._on_result,
                self, ids, x, y_real, output_names)
            client.async_infer(model_name=self._model_name, model_version=
                self._model_version, inputs=infer_inputs, outputs=
                outputs_req, callback=callback)
            self._num_waiting_for += 1
    with self._sync:

        def _all_processed():
            LOGGER.debug(f'wait for {self._num_waiting_for} unprocessed jobs')
            return self._num_waiting_for == 0
        self._processed_all = self._sync.wait_for(_all_processed, self.
            DEFAULT_MAX_FINISH_WAIT_S)
        if not self._processed_all:
            error_msg = (
                f'Runner {self._response_wait_t}s timeout received while waiting for results from server'
                )
            self._errors.append(error_msg)
    LOGGER.debug('Finished request thread')
