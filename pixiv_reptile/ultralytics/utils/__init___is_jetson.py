def is_jetson() ->bool:
    """
    Determines if the Python environment is running on a Jetson Nano or Jetson Orin device by checking the device model
    information.

    Returns:
        (bool): True if running on a Jetson Nano or Jetson Orin, False otherwise.
    """
    return 'NVIDIA' in PROC_DEVICE_MODEL
