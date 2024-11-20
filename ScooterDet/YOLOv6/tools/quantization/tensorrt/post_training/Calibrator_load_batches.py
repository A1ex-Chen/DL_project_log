def load_batches(self):
    for index in range(0, len(self.files), self.batch_size):
        for offset in range(self.batch_size):
            if self.use_cv2:
                image = cv2.imread(self.files[index + offset])
            else:
                image = Image.open(self.files[index + offset])
            self.batch[offset] = self.preprocess_func(image, *self.input_shape)
        logger.info('Calibration images pre-processed: {:}/{:}'.format(
            index + self.batch_size, len(self.files)))
        yield self.batch
