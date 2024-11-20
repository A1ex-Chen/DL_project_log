import logging
import select
import struct
from threading import Thread

from deepview_profile.io.sentinel import Sentinel

logger = logging.getLogger(__name__)


class Connection:
    """
    Manages an open connection to a client.

    This class must be constructed with an already-connected
    socket. Upon receipt of a message on the socket, the
    handler_function will be called with the raw message.

    Socket communication is performed using length-prefixed
    binary protobuf messages.

    The stop function must be called to close the connection.
    """





    @property

    @property


class ConnectionState:

