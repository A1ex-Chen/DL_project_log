def send_throughput_response(self, throughput, context):
    self._send_message(throughput, 'throughput', context)
