def visible_keypoints_in_image(dic):
    annotations = dic['annotations']
    return sum((np.array(ann['keypoints'][2::3]) > 0).sum() for ann in
        annotations if 'keypoints' in ann)
