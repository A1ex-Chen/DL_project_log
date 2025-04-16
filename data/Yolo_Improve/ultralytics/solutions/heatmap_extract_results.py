def extract_results(self, tracks):
    """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
    if tracks[0].boxes.id is not None:
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.tolist()
        self.track_ids = tracks[0].boxes.id.int().tolist()
