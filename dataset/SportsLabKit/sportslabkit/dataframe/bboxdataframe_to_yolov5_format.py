def to_yolov5_format(self, mapping: (dict[dict[Any, Any], dict[Any, Any]] |
    None)=None, na_class: int=0, h: (int | None)=None, w: (int | None)=None,
    save_dir: (str | None)=None):
    """Convert a dataframe to the YOLOv5 format.

        Converts a dataframe to the YOLOv5 format. The specification for each line is as follows:
        <class_id> <x_center> <y_center> <width> <height>

        * One row per object
        * Each row is class x_center y_center width height format.
        * Box coordinates must be normalized by the dimensions of the image (i.e. have values between 0 and 1)
        * Class numbers are zero-indexed (start from 0).

        Args:
            mapping (dict, optional): Mappings from team_id and player_id to class_id. Should contain one or two nested dictionaries like {'TeamID':{0:1}, 'PlayerID':{0:1}}. Defaults to None. If None,the class_id will be inferred from the team_id and player_id and set such that players=0 and ball=1.
            na_class (int, optional): Class ID for NaN values. Defaults to 0.
            h (int, optional): Height of the image. Unnecessary if the dataframe has height metadata. Defaults to None.
            w (int, optional): Width of the image. Unnecessary if the dataframe has width metadata. Defaults to None.
            save_dir (str, optional): If specified, saves a text file for each frame in the specified directory. Defaults to None.
        Returns:
            list: list of shape (N, M, 5) in YOLOv5 format. Where N is the number of frames, M is the number of objects in the frame, and 5 is the number of attributes (class_id, x_center, y_center, width, height).
        """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    df = self.to_long_df().reset_index()
    if mapping is None:
        mapping = {'TeamID': {'3': 1}, 'PlayerID': {}}
    team_mappings = df['TeamID'].map(mapping['TeamID'])
    player_mappings = df['PlayerID'].map(mapping['PlayerID'])
    df['class'] = player_mappings.combine_first(team_mappings).fillna(na_class
        ).astype(int)
    df['x'] = df['bb_left'] + df['bb_width'] / 2
    df['y'] = df['bb_top'] + df['bb_height'] / 2
    return_values = []
    groups = df.groupby('frame')
    for frame_num, group in groups:
        vals = group[['class', 'x', 'y', 'bb_width', 'bb_height']].values
        vals /= np.array([1, w, h, w, h])
        return_values.append(vals)
        if save_dir is not None:
            filename = f'{frame_num:06d}.txt'
            save_path = save_dir / filename
            np.savetxt(save_path, vals, fmt='%d %f %f %f %f')
    return return_values
