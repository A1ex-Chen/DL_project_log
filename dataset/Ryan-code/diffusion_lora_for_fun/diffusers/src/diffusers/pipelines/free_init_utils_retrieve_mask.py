def retrieve_mask(x):
    return 1 if x <= spatial_stop_frequency * 2 else 0
