def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str
    ) ->Tuple[np.ndarray, bool]:
    """Remove small disconnected regions or holes in a mask, returning the mask and a modification indicator."""
    import cv2
    assert mode in {'holes', 'islands'}, f'Provided mode {mode} is invalid'
    correct_holes = mode == 'holes'
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask
        , 8)
    sizes = stats[:, -1][1:]
    small_regions = [(i + 1) for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [
            int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True
