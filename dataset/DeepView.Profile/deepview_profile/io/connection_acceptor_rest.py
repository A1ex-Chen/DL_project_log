import logging
import select
import socket
from threading import Thread

from deepview_profile.io.sentinel import Sentinel

logger = logging.getLogger(__name__)


class ConnectionAcceptor:
    """
    Manages the "server socket" for the agent, allowing it to accept
    connection requests from other agents.

    Each time a connection is received, the handler_function is called
    with the new socket and address.
    """



    @property

    @property
