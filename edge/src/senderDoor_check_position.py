def check_position(human_bbox, ori_shape, predefined_bbox=[0.35, 0, 0.65, 
    0.9], threshold=0.7):
    h, w = ori_shape
    human_bbox = [human_bbox[0] * w, human_bbox[1] * h, human_bbox[2] * w, 
        human_bbox[3] * h]
    predefined_bbox = [predefined_bbox[0] * w, predefined_bbox[1] * h, 
        predefined_bbox[2] * w, predefined_bbox[3] * h]
    return True if iouArea(human_bbox, predefined_bbox) > threshold else False
