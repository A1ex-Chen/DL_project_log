def next_batch(self):
    if self.index < self.length:
        for i in range(self.batch_size):
            assert os.path.exists(self.img_list[i + self.index * self.
                batch_size]
                ), f'{self.img_list[i + self.index * self.batch_size]} not found!!'
            img = cv2.imread(self.img_list[i + self.index * self.batch_size])
            img = process_image(img, [self.input_h, self.input_w], 32)
            self.calibration_data[i] = img
        self.index += 1
        return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
    else:
        return np.array([])
