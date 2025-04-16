def _get_sparse_flow(gray_next, gray_prvs, prev_pts=None, max_points=300,
    min_distance=15, block_size=3, mask=None, quality_level=0.01):
    if prev_pts is None:
        prev_pts = cv2.goodFeaturesToTrack(gray_prvs, maxCorners=max_points,
            qualityLevel=quality_level, minDistance=min_distance, blockSize
            =block_size, mask=mask)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_prvs, gray_next,
        prev_pts, None)
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx].reshape((-1, 2))
    curr_pts = curr_pts[idx].reshape((-1, 2))
    return curr_pts, prev_pts
