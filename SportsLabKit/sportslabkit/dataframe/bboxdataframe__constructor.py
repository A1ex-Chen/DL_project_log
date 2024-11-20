@property
def _constructor(self: pd.DataFrame) ->type[BBoxDataFrame]:
    """Return the constructor for the DataFrame.

        Args:
            self (pd.DataFrame): DataFrame object.

        Returns:
            BBoxDataFrame: BBoxDataFrame object.
        """
    return BBoxDataFrame
