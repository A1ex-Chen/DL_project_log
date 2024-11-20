def start_counting(self, im0, tracks):
    """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
    self.im0 = im0
    self.extract_and_process_tracks(tracks)
    if self.view_img:
        self.display_frames()
    return self.im0
