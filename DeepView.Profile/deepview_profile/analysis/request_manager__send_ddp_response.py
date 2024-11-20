def _send_ddp_response(self, ddp_response, context):
    try:
        self._message_sender.send_ddp_response(ddp_response, context)
    except Exception:
        logger.exception('Exception occurred when sending ddp response.')
