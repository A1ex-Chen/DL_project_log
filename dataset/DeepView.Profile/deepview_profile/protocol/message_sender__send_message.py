def _send_message(self, message, payload_name, context):
    try:
        connection = self._connection_manager.get_connection(context.address)
        enclosing_message = pm.FromServer()
        getattr(enclosing_message, payload_name).CopyFrom(message)
        enclosing_message.sequence_number = context.sequence_number
        connection.send_bytes(enclosing_message.SerializeToString())
    except NoConnectionError:
        logger.debug(
            'Not sending message to (%s:%d) because it is no longer connected.'
            , *context.address)
