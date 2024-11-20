def to_codf(self: BBoxDataFrame, H: np.ndarray, method: str='bottom_middle'
    ) ->CoordinatesDataFrame:
    """
        Converts bounding box dataframe to a new coordinate dataframe using a given homography matrix.

        This function takes a dataframe of bounding boxes and applies a perspective transformation
        to a specified point within each bounding box (e.g., center, bottom middle, top middle) into
        a new coordinate frame (e.g., a pitch coordinate frame). The result is returned as a
        CoordinatesDataFrame.

        Args:
            self (BBoxDataFrame): A dataframe containing bounding box coordinates.
            H (np.ndarray): A 3x3 homography matrix used for the perspective transformation.
            method (str): Method to determine the point within the bounding box to transform.
                        Options include 'center', 'bottom_middle', 'top_middle'.

        Returns:
            CoordinatesDataFrame: A dataframe containing the transformed coordinates.

        Example:
            H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            bbox_data = BBoxDataFrame(...)
            codf = bbox_data.to_codf(H, method='bottom_middle')
        """
    assert H.shape == (3, 3), 'H must be a 3x3 matrix'
    long_df = self.to_long_df()
    if method == 'center':
        long_df['x'] = long_df['bb_left'] + long_df['bb_width'] / 2
        long_df['y'] = long_df['bb_top'] + long_df['bb_height'] / 2
    elif method == 'bottom_middle':
        long_df['x'] = long_df['bb_left'] + long_df['bb_width'] / 2
        long_df['y'] = long_df['bb_top'] + long_df['bb_height']
    elif method == 'top_middle':
        long_df['x'] = long_df['bb_left'] + long_df['bb_width'] / 2
        long_df['y'] = long_df['bb_top']
    else:
        raise ValueError(
            "Invalid method. Options are 'center', 'bottom_middle', 'top_middle'."
            )
    pts = long_df[['x', 'y']].values
    pitch_pts = cv2.perspectiveTransform(np.asarray([pts], dtype=np.float32), H
        )
    long_df['x'] = pitch_pts[0, :, 0]
    long_df['y'] = pitch_pts[0, :, 1]
    codf = CoordinatesDataFrame(long_df[['x', 'y']].unstack(level=['TeamID',
        'PlayerID']).reorder_levels([1, 2, 0], axis=1).sort_index(axis=1))
    return codf
