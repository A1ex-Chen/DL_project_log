def __init__(self, enqueue_response, message_sender, connection_manager):
    self._executor = ThreadPoolExecutor(max_workers=1)
    self._enqueue_response = enqueue_response
    self._message_sender = message_sender
    self._connection_manager = connection_manager
    self._nvml = NVML()
