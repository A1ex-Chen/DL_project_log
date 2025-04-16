def is_request_current(self, request):
    return request.sequence_number >= self.sequence_number
