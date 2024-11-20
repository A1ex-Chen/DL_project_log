def load_bbox(filename: PathLike) ->BBoxDataFrame:
    """Load a BBoxDataFrame from a file.

    Args:
        filename(PathLike): Path to bounding box file.

    Returns:
        bbox(BBoxDataFrame): BBoxDataFrame loaded from the file.
    """
    df_format = infer_bbox_format(filename)
    df = BBoxDataFrame(get_bbox_loader(df_format)(filename))
    df.rename_axis(['TeamID', 'PlayerID', 'Attributes'], axis=1, inplace=True)
    df.index.rename('frame', inplace=True)
    return df
