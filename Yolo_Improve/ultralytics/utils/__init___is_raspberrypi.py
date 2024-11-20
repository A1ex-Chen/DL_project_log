def is_raspberrypi() ->bool:
    """
    Determines if the Python environment is running on a Raspberry Pi by checking the device model information.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return 'Raspberry Pi' in PROC_DEVICE_MODEL
