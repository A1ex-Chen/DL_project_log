def inference(self, batched_inputs: List[Dict[str, torch.Tensor]],
    detected_instances: Optional[List[Instances]]=None, do_postprocess:
    bool=True):
    """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
    assert not self.training
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    if detected_instances is None:
        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [x['proposals'].to(self.device) for x in batched_inputs
                ]
        results, _ = self.roi_heads(images, features, proposals, None)
    else:
        detected_instances = [x.to(self.device) for x in detected_instances]
        results = self.roi_heads.forward_with_given_boxes(features,
            detected_instances)
    if do_postprocess:
        assert not torch.jit.is_scripting(
            ), 'Scripting is not supported for postprocess.'
        return GeneralizedRCNN._postprocess(results, batched_inputs, images
            .image_sizes)
    else:
        return results
