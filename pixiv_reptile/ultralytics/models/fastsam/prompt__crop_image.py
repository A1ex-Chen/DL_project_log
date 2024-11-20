def _crop_image(self, format_results):
    """Crops an image based on provided annotation format and returns cropped images and related data."""
    image = Image.fromarray(cv2.cvtColor(self.results[0].orig_img, cv2.
        COLOR_BGR2RGB))
    ori_w, ori_h = image.size
    annotations = format_results
    mask_h, mask_w = annotations[0]['segmentation'].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []
    for _, mask in enumerate(annotations):
        if np.sum(mask['segmentation']) <= 100:
            filter_id.append(_)
            continue
        bbox = self._get_bbox_from_mask(mask['segmentation'])
        cropped_boxes.append(self._segment_image(image, bbox))
        cropped_images.append(bbox)
    return cropped_boxes, cropped_images, not_crop, filter_id, annotations
