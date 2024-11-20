import logging
import signal
import threading

from deepview_profile.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)

logger = logging.getLogger(__name__)






    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    with SkylineServer(args.host, args.port) as server:
        _, port = server.listening_on
        logger.info(
            "DeepView interactive profiling session started! "
            "Listening on port %d.",
            port,
        )

        # Run the server until asked to terminate
        should_shutdown.wait()


def main(args):
    check_skyline_preconditions(args)
    initialize_skyline(args)
    actual_main(args)