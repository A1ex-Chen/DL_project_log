def inference(self, im, bboxes=None, points=None, labels=None, masks=None,
    multimask_output=False, *args, **kwargs):
    """
        Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder, and
        mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions shape (N,H,W). For SAM H=W=256.
            multimask_output (bool, optional): Flag to return multiple masks. Helpful for ambiguous prompts.

        Returns:
            (tuple): Contains the following three elements.
                - np.ndarray: The output masks in shape CxHxW, where C is the number of generated masks.
                - np.ndarray: An array of length C containing quality scores predicted by the model for each mask.
                - np.ndarray: Low-resolution logits of shape CxHxW for subsequent inference, where H=W=256.
        """
    bboxes = self.prompts.pop('bboxes', bboxes)
    points = self.prompts.pop('points', points)
    masks = self.prompts.pop('masks', masks)
    if all(i is None for i in [bboxes, points, masks]):
        return self.generate(im, *args, **kwargs)
    return self.prompt_inference(im, bboxes, points, labels, masks,
        multimask_output)
