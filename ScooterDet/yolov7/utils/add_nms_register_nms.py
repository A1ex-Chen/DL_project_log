def register_nms(self, *, score_thresh: float=0.25, nms_thresh: float=0.45,
    detections_per_img: int=100):
    """
        Register the ``EfficientNMS_TRT`` plugin node.
        NMS expects these shapes for its input tensors:
            - box_net: [batch_size, number_boxes, 4]
            - class_net: [batch_size, number_boxes, number_labels]
        Args:
            score_thresh (float): The scalar threshold for score (low scoring boxes are removed).
            nms_thresh (float): The scalar threshold for IOU (new boxes that have high IOU
                overlap with previously selected boxes are removed).
            detections_per_img (int): Number of best detections to keep after NMS.
        """
    self.infer()
    op_inputs = self.graph.outputs
    op = 'EfficientNMS_TRT'
    attrs = {'plugin_version': '1', 'background_class': -1,
        'max_output_boxes': detections_per_img, 'score_threshold':
        score_thresh, 'iou_threshold': nms_thresh, 'score_activation': 
        False, 'box_coding': 0}
    if self.precision == 'fp32':
        dtype_output = np.float32
    elif self.precision == 'fp16':
        dtype_output = np.float16
    else:
        raise NotImplementedError(
            f'Currently not supports precision: {self.precision}')
    output_num_detections = gs.Variable(name='num_dets', dtype=np.int32,
        shape=[self.batch_size, 1])
    output_boxes = gs.Variable(name='det_boxes', dtype=dtype_output, shape=
        [self.batch_size, detections_per_img, 4])
    output_scores = gs.Variable(name='det_scores', dtype=dtype_output,
        shape=[self.batch_size, detections_per_img])
    output_labels = gs.Variable(name='det_classes', dtype=np.int32, shape=[
        self.batch_size, detections_per_img])
    op_outputs = [output_num_detections, output_boxes, output_scores,
        output_labels]
    self.graph.layer(op=op, name='batched_nms', inputs=op_inputs, outputs=
        op_outputs, attrs=attrs)
    LOGGER.info(f"Created NMS plugin '{op}' with attributes: {attrs}")
    self.graph.outputs = op_outputs
    self.infer()
