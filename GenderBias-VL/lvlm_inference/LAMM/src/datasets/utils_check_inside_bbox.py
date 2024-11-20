def check_inside_bbox(keypoints_list, bbox):
    for keypoints in keypoints_list:
        x = keypoints[0]
        y = keypoints[1]
        if x > bbox[0] and x < bbox[2] and y > bbox[1] and y < bbox[3]:
            return True
    return False
