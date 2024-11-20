def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
    """
        Normalize, pad and batch the input images.
        """
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.backbone.size_divisibility)
    return images
