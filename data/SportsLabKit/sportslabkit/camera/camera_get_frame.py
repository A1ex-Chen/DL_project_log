def get_frame(self, frame_idx: int) ->np.ndarray:
    """Get frame from video.

        Args:
            frame (int): frame

        Returns:
            np.ndarray: frame
        """
    return self[frame_idx]
