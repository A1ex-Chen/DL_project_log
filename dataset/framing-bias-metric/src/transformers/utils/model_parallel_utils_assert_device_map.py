def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))
    device_map_blocks = [item for sublist in list(device_map.values()) for
        item in sublist]
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]
    assert len(duplicate_blocks
        ) == 0, 'Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These attention blocks were specified more than once: ' + str(
        duplicate_blocks)
    assert len(missing_blocks
        ) == 0, 'There are attention blocks for this model that are not specified in the device_map. Add these attention blocks to a device on the device_map: ' + str(
        missing_blocks)
    assert len(extra_blocks
        ) == 0, 'The device_map contains more attention blocks than this model has. Remove these from the device_map:' + str(
        extra_blocks)
