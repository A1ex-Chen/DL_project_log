def submit_request(self, analysis_request, context):
    if analysis_request.mock_response:
        self._handle_mock_analysis_request(analysis_request, context)
        return
    self._executor.submit(self._handle_analysis_request, analysis_request,
        context)
