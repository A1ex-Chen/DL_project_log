def to_mot_format(self):
    """Convert a dataframe to the MOT format.

        Returns:
            pd.DataFrame: Dataframe in MOT format.
        """
    df = self.to_long_df()
    df.reset_index(inplace=True)
    team_ids = df.TeamID.unique()
    player_ids = {team_id: df[df.TeamID == team_id].PlayerID.unique() for
        team_id in team_ids}
    id_map = {}
    num_ids = 0
    for team_id in team_ids:
        id_map[team_id] = {}
        for player_id in player_ids[team_id]:
            id_map[team_id][player_id] = num_ids
            num_ids += 1
    logger.debug(f'Using the following id_map: {id_map}')
    df['id'] = df.apply(lambda row: id_map[row['TeamID']][row['PlayerID']],
        axis=1)
    df = df[['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
        'conf']]
    df = df.assign(x=-1, y=-1, z=-1)
    df = df.sort_values(by=['frame', 'id'])
    dupes = df[df.duplicated(subset=['frame', 'id'])]
    if not dupes.empty:
        raise ValueError(
            f'Duplicate ids found in the following frames: {dupes.frame.unique()}'
            )
    return df
