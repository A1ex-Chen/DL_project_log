def box_prompt(self, bbox):
    """Modifies the bounding box properties and calculates IoU between masks and bounding box."""
    if self.results[0].masks is not None:
        assert bbox[2] != 0 and bbox[3
            ] != 0, 'Bounding box width and height should not be zero'
        masks = self.results[0].masks.data
        target_height, target_width = self.results[0].orig_shape
        h = masks.shape[1]
        w = masks.shape[2]
        if h != target_height or w != target_width:
            bbox = [int(bbox[0] * w / target_width), int(bbox[1] * h /
                target_height), int(bbox[2] * w / target_width), int(bbox[3
                ] * h / target_height)]
        bbox[0] = max(round(bbox[0]), 0)
        bbox[1] = max(round(bbox[1]), 0)
        bbox[2] = min(round(bbox[2]), w)
        bbox[3] = min(round(bbox[3]), h)
        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]],
            dim=(1, 2))
        orig_masks_area = torch.sum(masks, dim=(1, 2))
        union = bbox_area + orig_masks_area - masks_area
        iou = masks_area / union
        max_iou_index = torch.argmax(iou)
        self.results[0].masks.data = torch.tensor(np.array([masks[
            max_iou_index].cpu().numpy()]))
    return self.results
