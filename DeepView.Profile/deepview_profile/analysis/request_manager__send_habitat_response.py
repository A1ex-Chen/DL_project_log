def _send_habitat_response(self, habitat_resp, context):
    try:
        self._message_sender.send_habitat_response(habitat_resp, context)
    except Exception:
        logger.exception(
            'Exception occurred when sending a DeepView.Predict response.')
