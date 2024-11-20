def get_orig_size(json):
    if use_fix_crop_and_size:
        return resolution, resolution
    else:
        return int(json.get(WDS_JSON_WIDTH, 0.0)), int(json.get(
            WDS_JSON_HEIGHT, 0.0))
