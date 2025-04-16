def _handle_initialize_request(self, message, context):
    if context.state.initialized:
        self._message_sender.send_protocol_error(pm.ProtocolError.ErrorCode
            .ALREADY_INITIALIZED_CONNECTION, context)
        return
    if message.protocol_version != 5:
        self._message_sender.send_protocol_error(pm.ProtocolError.ErrorCode
            .UNSUPPORTED_PROTOCOL_VERSION, context)
        self._connection_manager.remove_connection(context.address)
        logger.error(
            'DeepView is out of date. Please update to the latest versions of the DeepView command line interface and plugin.'
            )
        return
    if not _validate_paths(message.project_root, message.entry_point):
        self._message_sender.send_protocol_error(pm.ProtocolError.ErrorCode
            .UNSUPPORTED_PROTOCOL_VERSION, context)
        self._connection_manager.remove_connection(context.address)
        logger.error('Invalid project root or entry point.')
        return
    logger.info('Connection addr:(%s:%d)', *context.address)
    logger.info('Project Root:   %s', message.project_root)
    logger.info('Entry Point:    %s', message.entry_point)
    self._connection_manager.get_connection(context.address).set_project_paths(
        message.project_root, message.entry_point)
    context.state.initialized = True
    self._message_sender.send_initialize_response(context)
