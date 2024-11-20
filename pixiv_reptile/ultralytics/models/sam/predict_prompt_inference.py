def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks
    =None, multimask_output=False):
    """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

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
    features = self.model.image_encoder(im
        ) if self.features is None else self.features
    src_shape, dst_shape = self.batch[1][0].shape[:2], im.shape[2:]
    r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], 
        dst_shape[1] / src_shape[1])
    if points is not None:
        points = torch.as_tensor(points, dtype=torch.float32, device=self.
            device)
        points = points[None] if points.ndim == 1 else points
        if labels is None:
            labels = np.ones(points.shape[0])
        labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
        points *= r
        points, labels = points[:, None, :], labels[:, None]
    if bboxes is not None:
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.
            device)
        bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
        bboxes *= r
    if masks is not None:
        masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
    points = (points, labels) if points is not None else None
    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=
        points, boxes=bboxes, masks=masks)
    pred_masks, pred_scores = self.model.mask_decoder(image_embeddings=
        features, image_pe=self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings
        =dense_embeddings, multimask_output=multimask_output)
    return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)
