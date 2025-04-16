def _send_throughput_response(self, throughput, context):
    try:
        self._message_sender.send_throughput_response(throughput, context)
    except Exception:
        logger.exception(
            'Exception occurred when sending a throughput response.')
