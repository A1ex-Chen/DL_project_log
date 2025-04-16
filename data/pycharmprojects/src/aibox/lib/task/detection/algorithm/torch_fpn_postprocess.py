def postprocess(self, result: List[Dict[str, Tensor]], image_shapes: List[
    Tuple[int, int]], original_image_sizes: List[Tuple[int, int]]) ->List[Dict
    [str, Tensor]]:
    if self.training:
        return result
    for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes,
        original_image_sizes)):
        boxes = pred['boxes']
        boxes = resize_boxes(boxes, im_s, o_im_s)
        result[i]['boxes'] = boxes
    return result
