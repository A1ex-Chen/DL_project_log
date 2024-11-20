def __init__(self, message_handler, closed_handler):
    self._connections = {}
    self._message_handler = message_handler
    self._closed_handler = closed_handler
