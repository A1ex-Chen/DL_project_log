def send_protocol_error(self, error_code, context):
    message = pm.ProtocolError()
    message.error_code = error_code
    self._send_message(message, 'error', context)
