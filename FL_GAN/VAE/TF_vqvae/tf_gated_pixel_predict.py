def predict(self, images):
    """
        images # shape [N,H,W,C]
        returns predicted image # shape [N,H,W,C]
        """
    pixel_value_probabilities = self.sess.run(self.output, {self.inputs:
        images})
    pixel_value_indices = np.argmax(pixel_value_probabilities, 4)
    pixel_values = np.multiply(pixel_value_indices, (self.pixel_depth - 1) /
        (self.q_levels - 1))
    return pixel_values
