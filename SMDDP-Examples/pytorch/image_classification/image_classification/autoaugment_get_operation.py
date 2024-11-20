def get_operation(self, method, magnitude_idx):
    magnitude = self.ranges[method][magnitude_idx]
    return lambda img: self.operations[method](img, magnitude)
