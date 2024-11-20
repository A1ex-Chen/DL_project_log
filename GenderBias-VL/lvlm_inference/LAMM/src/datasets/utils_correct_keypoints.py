def correct_keypoints(keypoints_list, gt_joint, thr=0.1):
    for keypoint in keypoints_list:
        if point_distance(np.array(keypoint), gt_joint) < thr:
            return True
    return False
