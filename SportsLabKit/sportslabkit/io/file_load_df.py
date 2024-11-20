def load_df(filename: PathLike, df_type: str='bbox') ->(BBoxDataFrame |
    CoordinatesDataFrame):
    """Loads either a BBoxDataFrame or a CoordinatesDataFrame from a file.

    Args:
        filename(Uinon[str, os.PathLike[Any]]): Path to file.
        df_type(str): Type of dataframe to load. Either 'bbox' or 'coordinates'.

    Returns:
        dataframe(Union[BBoxDataFrame, CoordinatesDataFrame]): DataFrame of file.
    """
    if df_type == 'bbox':
        df = load_bbox(filename)
    elif df_type == 'coordinates':
        df = load_codf(filename)
    else:
        raise ValueError(
            f"Unknown dataframe type {df_type}, must be 'bbox' or 'coordinates'"
            )
    return df
