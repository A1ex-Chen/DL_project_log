def post_process_sample(pred_bbox):
    pred_coor = cxcywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    return np.concatenate([pred_coor, scores[:, np.newaxis], classes[:, np.
        newaxis]], axis=-1)
