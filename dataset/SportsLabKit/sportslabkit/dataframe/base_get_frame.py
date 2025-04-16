def get_frame(self, frame):
    """Get a specific frame from the dataframe.

        Args:
            frame (int): Frame to get.

        Returns:
            pd.DataFrame: Dataframe with the frame.
        """
    if self.is_long_format():
        return self.xs(frame, level='frame')
    return self[self.index == frame]
