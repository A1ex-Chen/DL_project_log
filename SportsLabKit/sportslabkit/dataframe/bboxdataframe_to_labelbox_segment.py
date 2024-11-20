def to_labelbox_segment(self: BBoxDataFrame) ->dict:
    """Convert a dataframe to the Labelbox segment format.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.

        Returns:
            segment: Dictionary in Labelbox segment format.

        Notes:
            The Labelbox segment format is a dictionary with the following structure:
            {feature_name:
                {keyframes:
                    {frame:
                        {bbox:
                            {top: XX,
                            left: XX,
                            height: XX,
                            width: XX},
                        label: label
                        }
                    },
                    {frame:
                    ...

                    }
                }
            }
        """
    segment = {}
    for (team_id, player_id), player_bbdf in self.iter_players():
        feature_name = f'{team_id}_{player_id}'
        key_frames_dict = {}
        key_frames_dict['keyframes'] = []
        missing_bbox = 0
        for idx, row in player_bbdf.iterrows():
            try:
                key_frames_dict['keyframes'].append({'frame': idx + 1,
                    'bbox': {'top': int(row['bb_top']), 'left': int(row[
                    'bb_left']), 'height': int(row['bb_height']), 'width':
                    int(row['bb_width'])}})
            except ValueError:
                missing_bbox += 1
        if missing_bbox > 0:
            logger.debug(
                f'Missing {missing_bbox} bounding boxes for {feature_name}')
        segment[feature_name] = [key_frames_dict]
    return segment
