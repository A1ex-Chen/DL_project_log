def _send_analysis_error(self, exception, context):
    try:
        self._message_sender.send_analysis_error(exception, context)
    except Exception:
        logger.exception('Exception occurred when sending an analysis error.')
