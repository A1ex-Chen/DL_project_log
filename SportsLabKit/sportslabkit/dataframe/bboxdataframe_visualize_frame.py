def visualize_frame(self: BBoxDataFrame, frame_idx: int, frame: np.ndarray,
    draw_frame_id: bool=False) ->np.ndarray:
    """Visualize the bounding box of the specified frame.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            frame_idx (int): Frame ID.
            frame (np.ndarray): Frame image.
            draw_frame_id (bool, optional): Whether to draw the frame ID. Defaults to False.
        Returns:
            frame(np.ndarray): Frame image with bounding box.
        """
    if frame_idx not in self.index:
        return frame
    frame_df = self.loc[self.index == frame_idx]
    for (team_id, player_id), player_df in frame_df.iter_players():
        if player_df.isnull().any(axis=None):
            logger.debug(
                f'NaN value found at frame {frame_idx}, team {team_id}, player {player_id}. Skipping...'
                )
            continue
        logger.debug(
            f'Visualizing frame {frame_idx}, team {team_id}, player {player_id}'
            )
        if frame_idx not in player_df.index:
            logger.debug(f'Frame {frame_idx} not found in player_df')
            continue
        player_df.loc[frame_idx, ['bb_left', 'bb_top', 'bb_width', 'bb_height']
            ]
        x1, y1, w, h = player_df.loc[frame_idx, ['bb_left', 'bb_top',
            'bb_width', 'bb_height']].values.astype(int)
        x2, y2 = x1 + w, y1 + h
        label = f'{team_id}_{player_id}'
        player_id_int = sum([int(x) for x in str(hash(player_id))[1:]])
        color = _COLOR_NAMES[hash(player_id_int) % len(_COLOR_NAMES)]
        logger.debug(
            f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, label: {label}, color: {color}'
            )
        frame = add_bbox_to_frame(frame, x1, y1, x2, y2, label, color)
    if draw_frame_id:
        frame = add_frame_id_to_frame(frame, frame_idx)
    return frame
