def _handle_analysis_request(self, analysis_request, context):
    print('handle_analysis_request: begin')
    start_time = time.perf_counter()
    try:
        logger.debug('Processing request %d from (%s:%d).', context.
            sequence_number, *context.address)
        connection = self._connection_manager.get_connection(context.address)
        analyzer = analyze_project(connection.project_root, connection.
            entry_point, self._nvml, analysis_request.ddp_analysis_request)
        if self._early_disconnection_error(context):
            return
        breakdown = next(analyzer)
        self._enqueue_response(self._send_breakdown_response, breakdown,
            context)
        if self._early_disconnection_error(context):
            return
        throughput = next(analyzer)
        self._enqueue_response(self._send_throughput_response, throughput,
            context)
        if self._early_disconnection_error(context):
            return
        habitat_resp = next(analyzer)
        self._enqueue_response(self._send_habitat_response, habitat_resp,
            context)
        if self._early_disconnection_error(context):
            return
        utilization_resp = next(analyzer)
        self._enqueue_response(self._send_utilization_response,
            utilization_resp, context)
        if self._early_disconnection_error(context):
            return
        energy_resp = next(analyzer)
        self._enqueue_response(self._send_energy_response, energy_resp, context
            )
        if self._early_disconnection_error(context):
            return
        if analysis_request.ddp_analysis_request:
            ddp_resp = next(analyzer)
            self._enqueue_response(self._send_ddp_response, ddp_resp, context)
        next(analyzer)
        elapsed_time = time.perf_counter() - start_time
        logger.debug(
            'Processed analysis request %d from (%s:%d) in %.4f seconds.',
            context.sequence_number, *context.address, elapsed_time)
    except AnalysisError as ex:
        self._enqueue_response(self._send_analysis_error, ex, context)
    except Exception:
        logger.exception('Exception occurred when handling analysis request.')
        self._enqueue_response(self._send_analysis_error, AnalysisError(
            'An unexpected error occurred when analyzing your model. Please file a bug report and then restart DeepView.'
            ), context)
