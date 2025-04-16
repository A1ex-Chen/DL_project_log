def close(self):
    """Close stream loader and release resources."""
    self.running = False
    for thread in self.threads:
        if thread.is_alive():
            thread.join(timeout=5)
    for cap in self.caps:
        try:
            cap.release()
        except Exception as e:
            LOGGER.warning(
                f'WARNING ⚠️ Could not release VideoCapture object: {e}')
    cv2.destroyAllWindows()
