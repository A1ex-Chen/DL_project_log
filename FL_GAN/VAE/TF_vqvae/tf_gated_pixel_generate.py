def generate(self, images):
    samples = images[0:9, :, :, :]
    occlude_start_row = 18
    samples[:, occlude_start_row:, :, :] = 0.0
    for i in range(occlude_start_row, self.height):
        for j in range(self.width):
            for k in range(self.channel):
                next_sample = self.predict(samples) / (self.pixel_depth - 1.0)
                samples[:, i, j, k] = next_sample[:, i, j, k]
    return samples
