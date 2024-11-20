def load_gps_from_yaml(yaml_path: str) ->CoordinatesDataFrame:
    """Load GPS data from a YAML file.

    Args:
        yaml_path(str): Path to yaml file.

    Returns:
        merged_dataframe(CoordinatesDataFrame): DataFrame of merged gpsports and statsports.
    """
    cfg = OmegaConf.load(yaml_path)
    playerids, teamids, filepaths = [], [], []
    for device in cfg.devices:
        playerids.append(device.playerid)
        teamids.append(device.teamid)
        filepaths.append(Path(device.filepath))
    return load_gps(filepaths, playerids, teamids)
