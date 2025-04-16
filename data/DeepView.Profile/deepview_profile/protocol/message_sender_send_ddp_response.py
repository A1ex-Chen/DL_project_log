def send_ddp_response(self, ddp_resp, context):
    self._send_message(ddp_resp, 'ddp', context)
