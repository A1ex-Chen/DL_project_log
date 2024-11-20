def setup_logger():
    logging.basicConfig(level=logging.INFO if dist_utils.is_main_process() else
        logging.WARN, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()])
