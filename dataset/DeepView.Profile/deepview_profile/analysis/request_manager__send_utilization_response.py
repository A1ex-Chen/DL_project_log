def _send_utilization_response(self, utilization_resp, context):
    try:
        self._message_sender.send_utilization_response(utilization_resp,
            context)
    except Exception:
        logger.exception(
            'Exception occurred when sending utilization response.')
