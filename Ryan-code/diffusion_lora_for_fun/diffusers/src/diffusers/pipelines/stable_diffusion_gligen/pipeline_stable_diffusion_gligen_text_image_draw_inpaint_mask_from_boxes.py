def draw_inpaint_mask_from_boxes(self, boxes, size):
    """
        Create an inpainting mask based on given boxes. This function generates an inpainting mask using the provided
        boxes to mark regions that need to be inpainted.
        """
    inpaint_mask = torch.ones(size[0], size[1])
    for box in boxes:
        x0, x1 = box[0] * size[0], box[2] * size[0]
        y0, y1 = box[1] * size[1], box[3] * size[1]
        inpaint_mask[int(y0):int(y1), int(x0):int(x1)] = 0
    return inpaint_mask
