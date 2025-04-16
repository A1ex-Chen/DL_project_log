def extract_tracks(self, tracks):
    """
        Extracts tracking results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
    self.boxes = tracks[0].boxes.xyxy.cpu()
    self.clss = tracks[0].boxes.cls.cpu().tolist()
    self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()
