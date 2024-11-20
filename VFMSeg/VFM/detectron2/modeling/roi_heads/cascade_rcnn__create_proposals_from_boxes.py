def _create_proposals_from_boxes(self, boxes, image_sizes):
    """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
    boxes = [Boxes(b.detach()) for b in boxes]
    proposals = []
    for boxes_per_image, image_size in zip(boxes, image_sizes):
        boxes_per_image.clip(image_size)
        if self.training:
            boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
        prop = Instances(image_size)
        prop.proposal_boxes = boxes_per_image
        proposals.append(prop)
    return proposals
