def _cat_labels(self, mosaic_labels):
    """Return labels with mosaic border instances clipped."""
    if len(mosaic_labels) == 0:
        return {}
    cls = []
    instances = []
    imgsz = self.imgsz * 2
    for labels in mosaic_labels:
        cls.append(labels['cls'])
        instances.append(labels['instances'])
    final_labels = {'im_file': mosaic_labels[0]['im_file'], 'ori_shape':
        mosaic_labels[0]['ori_shape'], 'resized_shape': (imgsz, imgsz),
        'cls': np.concatenate(cls, 0), 'instances': Instances.concatenate(
        instances, axis=0), 'mosaic_border': self.border}
    final_labels['instances'].clip(imgsz, imgsz)
    good = final_labels['instances'].remove_zero_area_boxes()
    final_labels['cls'] = final_labels['cls'][good]
    if 'texts' in mosaic_labels[0]:
        final_labels['texts'] = mosaic_labels[0]['texts']
    return final_labels
