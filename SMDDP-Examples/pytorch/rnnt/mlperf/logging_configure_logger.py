def configure_logger(output_dir, benchmark):
    mllog.config(filename=os.path.join(output_dir, f'{benchmark}.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
