def handle_message(self, raw_data, address):
    try:
        message = pm.FromClient()
        message.ParseFromString(raw_data)
        logger.debug('Received message from (%s:%d).', *address)
        state = self._connection_manager.get_connection_state(address)
        if not state.is_request_current(message):
            logger.debug('Ignoring stale message from (%s:%d).', *address)
            return
        state.update_sequence(message)
        message_type = message.WhichOneof('payload')
        if message_type is None:
            logger.warn('Received empty message from (%s:%d).', *address)
            return
        context = RequestContext(state=state, address=address,
            sequence_number=message.sequence_number)
        if message_type == 'initialize':
            self._handle_initialize_request(getattr(message, message_type),
                context)
        elif message_type == 'analysis':
            self._handle_analysis_request(getattr(message, message_type),
                context)
        else:
            raise AssertionError('Invalid message type "{}".'.format(
                message_type))
    except NoConnectionError:
        logger.debug(
            'Dropping message from (%s:%d) because it is no longer connected.',
            *address)
    except Exception:
        logger.exception(
            'Processing message from (%s:%d) resulted in an exception.', *
            address)
