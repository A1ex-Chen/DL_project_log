def to_bbdf(self) ->BBoxDataFrame:
    """Convert the tracker predictions to a BBoxDataFrame.

        Returns:
            BBoxDataFrame: BBoxDataFrame of the tracker
        """
    if self.global_step >= self.steps_alive:
        frame_range = range(self.global_step + 1 - self.steps_alive, self.
            global_step + 1)
    else:
        raise ValueError(
            f'Global step {self.global_step} is less than steps alive {self.steps_alive}'
            )
    data_dict = {'frame': list(frame_range), 'id': [self.id for _ in
        frame_range]}
    for observation in self._observations:
        if self.get_observation(observation) is not None:
            data_dict[observation] = self.get_observations(observation)
        elif observation == 'score':
            data_dict[observation] = [(1) for _ in frame_range]
    df = pd.DataFrame(data_dict)
    df = pd.DataFrame(df['box'].to_list(), columns=['bb_left', 'bb_top',
        'bb_width', 'bb_height']).join(df.drop(columns=['box']))
    df.rename(columns={'global_step': 'frame', 'score': 'conf'}, inplace=True)
    df.set_index(['frame'], inplace=True)
    if 'conf' not in df.columns:
        df['conf'] = 1.0
    box_df = df[['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']]
    team_id = self.team_id or 0
    player_id = self.player_id or df.id.unique()[0]
    idx = pd.MultiIndex.from_product([[team_id], [player_id], box_df.
        columns], names=['TeamID', 'PlayerID', 'Attributes'])
    bbdf = BBoxDataFrame(box_df.values, index=df.index, columns=idx)
    return bbdf
