import logging
from concurrent.futures import ThreadPoolExecutor

from deepview_profile.analysis.request_manager import AnalysisRequestManager
from deepview_profile.io.connection_acceptor import ConnectionAcceptor
from deepview_profile.io.connection_manager import ConnectionManager
from deepview_profile.protocol.message_handler import MessageHandler
from deepview_profile.protocol.message_sender import MessageSender

logger = logging.getLogger(__name__)


class SkylineServer:





    @property





        self._analysis_request_manager.stop()
        self._main_executor.submit(shutdown).result()
        self._main_executor.shutdown()
        logger.debug("DeepView server has shut down.")

    @property
    def listening_on(self):
        return (self._connection_acceptor.host, self._connection_acceptor.port)

    def _on_message(self, data, address):
        print("on_message:", data, address)
        # Do not call directly - called by a connection
        self._main_executor.submit(
            self._message_handler.handle_message,
            data,
            address,
        )

    def _on_new_connection(self, socket, address):
        print("on_new_connection", socket, address)
        # Do not call directly - called by _connection_acceptor
        self._main_executor.submit(
            self._connection_manager.register_connection,
            socket,
            address,
        )

    def _on_connection_closed(self, address):
        # Do not call directly - called by a connection when it is closed
        self._main_executor.submit(
            self._connection_manager.remove_connection,
            address,
        )

    def _submit_work(self, func, *args, **kwargs):
        print("submit_work", func)
        # print("submit_work args:", args)
        logger.debug("submit_work args:", args)
        print("submit_work kwargs:", kwargs)
        # Do not call directly - called by another thread to submit work
        # onto the main executor
        self._main_executor.submit(func, *args, **kwargs)