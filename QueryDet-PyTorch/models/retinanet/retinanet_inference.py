def inference(self, box_cls, box_delta, anchors, image_sizes):
    """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
    results = []
    times = []
    box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
    box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    for img_idx, image_size in enumerate(image_sizes):
        box_cls_per_image = [box_cls_per_level[img_idx] for
            box_cls_per_level in box_cls]
        box_reg_per_image = [box_reg_per_level[img_idx] for
            box_reg_per_level in box_delta]
        results_per_image = self.inference_single_image(box_cls_per_image,
            box_reg_per_image, anchors, (image_size[0] * self.scale_factor,
            image_size[1] * self.scale_factor))
        results.append(results_per_image)
    return results
