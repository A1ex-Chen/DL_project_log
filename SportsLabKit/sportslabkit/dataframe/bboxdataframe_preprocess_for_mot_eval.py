def preprocess_for_mot_eval(self):
    """Preprocess a dataframe for evaluation using the MOT metrics.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.

        Returns:
            ids (list): List of lists of object ids for each frame.
            dets (list): A list of arrays of detections in the format (x, y, w, h) for each frame.
        """
    if self.size == 0:
        return [], []
    list_of_tuples = self.to_list_of_tuples_format()
    list_of_list_of_bboxes = np.split(list_of_tuples, np.unique(
        list_of_tuples[:, IMAGE_NAME_INDEX], return_index=True)[1][1:])
    frame_idxs = []
    for list_of_bboxes in list_of_list_of_bboxes:
        try:
            frame_idxs.append(list_of_bboxes[:, IMAGE_NAME_INDEX].astype(
                'int64')[0])
        except IndexError:
            frame_idxs.append(None)
    ids = [list_of_bboxes[:, OBJECT_ID_INDEX].astype('int64') for
        list_of_bboxes in list_of_list_of_bboxes]
    dets = [list_of_bboxes[:, [X_INDEX, Y_INDEX, W_INDEX, H_INDEX]].astype(
        'int64') for list_of_bboxes in list_of_list_of_bboxes]
    start_frame = self.index.min()
    end_frame = self.index.max()
    missing_frames = np.setdiff1d(range(start_frame, end_frame + 1), frame_idxs
        )
    for missing_frame in missing_frames:
        insert_index = missing_frame - start_frame
        ids.insert(insert_index, np.array([]))
        dets.insert(insert_index, np.array([]))
    return ids, dets
