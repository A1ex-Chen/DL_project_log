def __init__(self, host, port):
    self._requested_host = host
    self._requested_port = port
    self._connection_acceptor = ConnectionAcceptor(self._requested_host,
        self._requested_port, self._on_new_connection)
    self._connection_manager = ConnectionManager(self._on_message, self.
        _on_connection_closed)
    self._message_sender = MessageSender(self._connection_manager)
    self._analysis_request_manager = AnalysisRequestManager(self.
        _submit_work, self._message_sender, self._connection_manager)
    self._message_handler = MessageHandler(self._connection_manager, self.
        _message_sender, self._analysis_request_manager)
    self._main_executor = ThreadPoolExecutor(max_workers=1)
