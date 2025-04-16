def write_model(self, dir_work, epoch):
    """
        Do the I/O, report stats
        dir_work: A PosixPath
        """
    model_file = dir_work / 'model.h5'
    self.debug("writing model to: '%s'" % self.relpath(model_file))
    start = time.time()
    self.model.save(model_file)
    stop = time.time()
    duration = stop - start
    stats = os.stat(model_file)
    MB = stats.st_size / (1024 * 1024)
    rate = MB / duration
    self.debug('model wrote: %0.3f MB in %0.3f seconds (%0.2f MB/s).' % (MB,
        duration, rate))
    self.checksum(dir_work)
    self.write_json(dir_work / 'ckpt-info.json', epoch)
