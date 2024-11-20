def _handle_analysis_request(self, message, context):
    if not context.state.initialized:
        self._message_sender.send_protocol_error(pm.ProtocolError.ErrorCode
            .UNINITIALIZED_CONNECTION, context)
        return
    self._analysis_request_manager.submit_request(message, context)
