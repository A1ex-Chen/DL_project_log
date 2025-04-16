def warmup(self):
    LOGGER.info('Warming up extracting model...')
    for i in range(10):
        im_tmp = np.random.randint(0, 255, (np.random.randint(20, 400), np.
            random.randint(20, 400), 3), 'uint8')
        self.extract(im_tmp)
