def forward(self, x):
    if self.input_is_batched:
        start_index = self.image_count
        end_index = self.image_count + len(x)
        self.image_count += len(x)
        if self.image_count >= len(self.precomputed_detections):
            self.reset_image_count()
        return self.precomputed_detections[start_index:end_index]
    else:
        detections = self.precomputed_detections[self.image_count]
        self.image_count += 1
        results = [detections]
        if self.image_count >= len(self.precomputed_detections):
            self.reset_image_count()
        return results
