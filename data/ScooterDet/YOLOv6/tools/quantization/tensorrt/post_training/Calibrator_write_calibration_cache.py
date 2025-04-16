def write_calibration_cache(self, cache):
    with open(self.cache_file, 'wb') as f:
        logger.info('Caching calibration data for future use: {:}'.format(
            self.cache_file))
        f.write(cache)
