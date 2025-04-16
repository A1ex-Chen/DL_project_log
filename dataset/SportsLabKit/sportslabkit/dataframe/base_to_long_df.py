def to_long_df(self, level='Attributes', dropna=True):
    """Convert a dataframe to a long format.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            level (str, optional): Level to convert to long format. Defaults to 'Attributes'. Options are 'Attributes', 'TeamID', 'PlayerID'.

        Returns:
            pd.DataFrame: Dataframe in long format.
        """
    df = self.copy()
    levels = ['TeamID', 'PlayerID', 'Attributes']
    levels.remove(level)
    df = df.stack(level=levels, dropna=dropna)
    return df
