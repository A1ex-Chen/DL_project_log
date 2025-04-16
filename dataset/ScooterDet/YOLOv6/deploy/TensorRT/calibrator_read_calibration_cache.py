def read_calibration_cache(self):
    if os.path.exists(self.cache_file):
        with open(self.cache_file, 'rb') as f:
            logger.info('Using calibration cache to save time: {:}'.format(
                self.cache_file))
            return f.read()
