@staticmethod
def symbolic(g, boxes, scores, background_class=-1, box_coding=1,
    iou_threshold=0.45, max_output_boxes=100, plugin_version='1',
    score_activation=0, score_threshold=0.25):
    out = g.op('TRT::EfficientNMS_TRT', boxes, scores, background_class_i=
        background_class, box_coding_i=box_coding, iou_threshold_f=
        iou_threshold, max_output_boxes_i=max_output_boxes,
        plugin_version_s=plugin_version, score_activation_i=
        score_activation, score_threshold_f=score_threshold, outputs=4)
    nums, boxes, scores, classes = out
    return nums, boxes, scores, classes
