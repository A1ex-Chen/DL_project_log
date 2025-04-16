def send_initialize_request(self, project_root):
    request = innpv_pb2.InitializeRequest()
    request.protocol_version = 5
    request.project_root = project_root
    request.entry_point = 'entry_point.py'
    self.send_message(request)
