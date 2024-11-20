def send_analysis_request(self):
    request = innpv_pb2.AnalysisRequest()
    request.mock_response = False
    self.send_message(request)
