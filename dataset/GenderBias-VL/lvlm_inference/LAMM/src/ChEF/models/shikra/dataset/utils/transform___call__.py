def __call__(self, image: Image.Image, labels: Dict[str, Any]=None) ->Tuple[
    Image.Image, Optional[Dict[str, Any]]]:
    width, height = image.size
    processed_image = expand2square(image, background_color=self.
        background_color)
    if labels is None:
        return processed_image, labels
    if 'boxes' in labels:
        bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in
            labels['boxes']]
        labels['boxes'] = bboxes
    if 'points' in labels:
        points = [point_xy_expand2square(point, w=width, h=height) for
            point in labels['points']]
        labels['points'] = points
    return processed_image, labels
