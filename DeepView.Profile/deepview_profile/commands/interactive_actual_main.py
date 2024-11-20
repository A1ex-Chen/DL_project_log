def actual_main(args):
    from deepview_profile.server import SkylineServer
    should_shutdown = threading.Event()

    def signal_handler(signal, frame):
        should_shutdown.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    with SkylineServer(args.host, args.port) as server:
        _, port = server.listening_on
        logger.info(
            'DeepView interactive profiling session started! Listening on port %d.'
            , port)
        should_shutdown.wait()
