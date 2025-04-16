def prepare(self, predictions, iou_type):
    if iou_type == 'bbox':
        return self.prepare_for_coco_detection(predictions)
    elif iou_type == 'segm':
        return self.prepare_for_coco_segmentation(predictions)
    elif iou_type == 'keypoints':
        return self.prepare_for_coco_keypoint(predictions)
    else:
        raise ValueError('Unknown iou type {}'.format(iou_type))
