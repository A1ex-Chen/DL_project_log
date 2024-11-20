def postprocess(self, preds, img, orig_imgs):
    """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (list): List of [predictions, extra] from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
    if not isinstance(preds, (list, tuple)):
        preds = [preds, None]
    nd = preds[0].shape[-1]
    bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for i, bbox in enumerate(bboxes):
        bbox = ops.xywh2xyxy(bbox)
        score, cls = scores[i].max(-1, keepdim=True)
        idx = score.squeeze(-1) > self.args.conf
        if self.args.classes is not None:
            idx = (cls == torch.tensor(self.args.classes, device=cls.device)
                ).any(1) & idx
        pred = torch.cat([bbox, score, cls], dim=-1)[idx]
        orig_img = orig_imgs[i]
        oh, ow = orig_img.shape[:2]
        pred[..., [0, 2]] *= ow
        pred[..., [1, 3]] *= oh
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, boxes=pred))
    return results
