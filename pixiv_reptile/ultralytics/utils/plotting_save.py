def save(self, filename='image.jpg'):
    """Save the annotated image to 'filename'."""
    cv2.imwrite(filename, np.asarray(self.im))
