def _early_disconnection_error(self, context):
    if not context.state.connected:
        logger.error(
            'Aborting request %d from (%s:%d) early because the client has disconnected.'
            , context.sequence_number, *context.address)
        return True
    return False
