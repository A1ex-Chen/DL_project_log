def get_data(self, frame=None, playerid=None, teamid=None, attributes=None):
    """Get specific data from the dataframe.

        Args:
            frame (int or list of int, optional): Frame(s) to get.
            player (int or list of int, optional): Player ID(s) to get.
            team (int or list of int, optional): Team ID(s) to get.
            attributes (str or list of str, optional): Attribute(s) to get.

        Returns:
            pd.DataFrame: Dataframe with the selected data.
        """
    df = self
    if frame is not None:
        df = df.iloc[[frame]]
    if playerid is not None:
        df = df.xs(playerid, level='PlayerID', axis=1, drop_level=False)
    if teamid is not None:
        df = df.xs(teamid, level='TeamID', axis=1, drop_level=False)
    if attributes is not None:
        df = df.xs(attributes, level='Attributes', axis=1, drop_level=False)
    return df
