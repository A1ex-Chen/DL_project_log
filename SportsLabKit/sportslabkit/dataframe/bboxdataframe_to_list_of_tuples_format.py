def to_list_of_tuples_format(self, mapping: (dict[dict[Any, Any], dict[Any,
    Any]] | None)=None, na_class: (int | str)='player'):
    """Convert a dataframe to a list of tuples.

        Converts a dataframe to a list of tuples necessary for calculating object detection metrics such as mAP and AP scores. The specification for each list element is as follows:
        (x, y, w, h, confidence, class_id, image_name, object_id)

        Returns:
            list: List of tuples.
        """
    long_df = self.to_long_df().reset_index()
    long_df = long_df.dropna()
    if mapping is None:
        mapping = {'TeamID': {'3': 'ball'}, 'PlayerID': {}}
    team_mappings = long_df['TeamID'].map(mapping['TeamID'])
    player_mappings = long_df['PlayerID'].map(mapping['PlayerID'])
    long_df['class'] = player_mappings.combine_first(team_mappings).fillna(
        na_class)
    long_df['image_name'] = long_df['frame'].astype(int)
    long_df['object_id'] = long_df['PlayerID'].astype(str) + '_' + long_df[
        'TeamID'].astype(str)
    assigned_ids = {p_id_t_id: object_id for object_id, p_id_t_id in
        enumerate(long_df['object_id'].unique())}
    long_df['object_id'] = long_df['object_id'].map(assigned_ids).astype(int)
    cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class',
        'image_name', 'object_id']
    return long_df[cols].values
