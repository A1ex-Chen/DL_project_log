def _send_energy_response(self, energy_resp, context):
    try:
        self._message_sender.send_energy_response(energy_resp, context)
    except Exception:
        logger.exception('Exception occurred when sending an energy response.')
