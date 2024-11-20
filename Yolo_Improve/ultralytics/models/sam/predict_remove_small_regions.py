@staticmethod
def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
    """
        Perform post-processing on segmentation masks generated by the Segment Anything Model (SAM). Specifically, this
        function removes small disconnected regions and holes from the input masks, and then performs Non-Maximum
        Suppression (NMS) to eliminate any newly created duplicate boxes.

        Args:
            masks (torch.Tensor): A tensor containing the masks to be processed. Shape should be (N, H, W), where N is
                                  the number of masks, H is height, and W is width.
            min_area (int): The minimum area below which disconnected regions and holes will be removed. Defaults to 0.
            nms_thresh (float): The IoU threshold for the NMS algorithm. Defaults to 0.7.

        Returns:
            (tuple([torch.Tensor, List[int]])):
                - new_masks (torch.Tensor): The processed masks with small regions removed. Shape is (N, H, W).
                - keep (List[int]): The indices of the remaining masks post-NMS, which can be used to filter the boxes.
        """
    import torchvision
    if len(masks) == 0:
        return masks
    new_masks = []
    scores = []
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8)
        mask, changed = remove_small_regions(mask, min_area, mode='holes')
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode='islands')
        unchanged = unchanged and not changed
        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        scores.append(float(unchanged))
    new_masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(new_masks)
    keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores),
        nms_thresh)
    return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep
