def mask_to_polygons(self, mask):
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.
        CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [(x + 0.5) for x in res if len(x) >= 6]
    return res, has_holes
