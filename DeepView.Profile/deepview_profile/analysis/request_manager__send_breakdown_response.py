def _send_breakdown_response(self, breakdown, context):
    try:
        self._message_sender.send_breakdown_response(breakdown, context)
    except Exception:
        logger.exception(
            'Exception occurred when sending a breakdown response.')
