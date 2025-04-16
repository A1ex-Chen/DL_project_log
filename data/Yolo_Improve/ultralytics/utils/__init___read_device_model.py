def read_device_model() ->str:
    """
    Reads the device model information from the system and caches it for quick access. Used by is_jetson() and
    is_raspberrypi().

    Returns:
        (str): Model file contents if read successfully or empty string otherwise.
    """
    with contextlib.suppress(Exception):
        with open('/proc/device-tree/model') as f:
            return f.read()
    return ''
