def _preprocess_image(self, image):
    """Convert the image to grayscale and apply thresholding and morphological operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    gray = gray.astype(np.uint8)
    kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray
