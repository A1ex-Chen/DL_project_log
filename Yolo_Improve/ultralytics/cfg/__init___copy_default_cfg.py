def copy_default_cfg():
    """Copy and create a new default configuration file with '_copy' appended to its name, providing usage example."""
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace('.yaml', '_copy.yaml'
        )
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"""{DEFAULT_CFG_PATH} copied to {new_file}
Example YOLO command with this new custom cfg:
    yolo cfg='{new_file}' imgsz=320 batch=8"""
        )
