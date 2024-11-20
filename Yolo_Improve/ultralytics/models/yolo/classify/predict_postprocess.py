def postprocess(self, preds, img, orig_imgs):
    """Post-processes predictions to return Results objects."""
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.
            names, probs=pred))
    return results
