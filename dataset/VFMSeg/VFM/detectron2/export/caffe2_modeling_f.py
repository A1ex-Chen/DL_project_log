def f(batched_inputs, c2_inputs, c2_results):
    _, im_info = c2_inputs
    image_sizes = [[int(im[0]), int(im[1])] for im in im_info]
    dummy_images = ImageList(torch.randn((len(im_info), 3) + tuple(
        image_sizes[0])), image_sizes)
    num_features = len([x for x in c2_results.keys() if x.startswith(
        'box_cls_')])
    pred_logits = [c2_results['box_cls_{}'.format(i)] for i in range(
        num_features)]
    pred_anchor_deltas = [c2_results['box_delta_{}'.format(i)] for i in
        range(num_features)]
    dummy_features = [x.clone()[:, 0:0, :, :] for x in pred_logits]
    self.num_classes = pred_logits[0].shape[1] // (pred_anchor_deltas[0].
        shape[1] // 4)
    results = self.forward_inference(dummy_images, dummy_features, [
        pred_logits, pred_anchor_deltas])
    return meta_arch.GeneralizedRCNN._postprocess(results, batched_inputs,
        image_sizes)
