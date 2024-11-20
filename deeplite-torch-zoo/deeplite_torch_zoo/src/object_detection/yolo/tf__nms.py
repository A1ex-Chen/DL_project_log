@staticmethod
def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):
    boxes, classes, scores = x
    class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
    scores_inp = tf.reduce_max(scores, -1)
    selected_inds = tf.image.non_max_suppression(boxes, scores_inp,
        max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=
        conf_thres)
    selected_boxes = tf.gather(boxes, selected_inds)
    padded_boxes = tf.pad(selected_boxes, paddings=[[0, topk_all - tf.shape
        (selected_boxes)[0]], [0, 0]], mode='CONSTANT', constant_values=0.0)
    selected_scores = tf.gather(scores_inp, selected_inds)
    padded_scores = tf.pad(selected_scores, paddings=[[0, topk_all - tf.
        shape(selected_boxes)[0]]], mode='CONSTANT', constant_values=-1.0)
    selected_classes = tf.gather(class_inds, selected_inds)
    padded_classes = tf.pad(selected_classes, paddings=[[0, topk_all - tf.
        shape(selected_boxes)[0]]], mode='CONSTANT', constant_values=-1.0)
    valid_detections = tf.shape(selected_inds)[0]
    return padded_boxes, padded_scores, padded_classes, valid_detections
