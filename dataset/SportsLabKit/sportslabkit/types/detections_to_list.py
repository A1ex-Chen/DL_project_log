def to_list(self):
    if len(self.preds) == 0:
        logger.debug('No results to show.')
        return []
    dets = []
    for x, y, w, h, conf, class_id in self.preds:
        det = Detection([x, y, w, h], conf, class_id)
        dets.append(det)
    return dets
