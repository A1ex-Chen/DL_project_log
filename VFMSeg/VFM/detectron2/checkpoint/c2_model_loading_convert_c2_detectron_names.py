def convert_c2_detectron_names(weights):
    """
    Map Caffe2 Detectron weight names to Detectron2 names.

    Args:
        weights (dict): name -> tensor

    Returns:
        dict: detectron2 names -> tensor
        dict: detectron2 names -> C2 names
    """
    logger = logging.getLogger(__name__)
    logger.info('Renaming Caffe2 weights ......')
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = convert_basic_c2_names(layer_keys)
    layer_keys = [k.replace('conv.rpn.fpn2',
        'proposal_generator.rpn_head.conv') for k in layer_keys]
    layer_keys = [k.replace('conv.rpn', 'proposal_generator.rpn_head.conv') for
        k in layer_keys]
    layer_keys = [k.replace('rpn.bbox.pred.fpn2',
        'proposal_generator.rpn_head.anchor_deltas') for k in layer_keys]
    layer_keys = [k.replace('rpn.cls.logits.fpn2',
        'proposal_generator.rpn_head.objectness_logits') for k in layer_keys]
    layer_keys = [k.replace('rpn.bbox.pred',
        'proposal_generator.rpn_head.anchor_deltas') for k in layer_keys]
    layer_keys = [k.replace('rpn.cls.logits',
        'proposal_generator.rpn_head.objectness_logits') for k in layer_keys]
    layer_keys = [re.sub('^bbox\\.pred', 'bbox_pred', k) for k in layer_keys]
    layer_keys = [re.sub('^cls\\.score', 'cls_score', k) for k in layer_keys]
    layer_keys = [re.sub('^fc6\\.', 'box_head.fc1.', k) for k in layer_keys]
    layer_keys = [re.sub('^fc7\\.', 'box_head.fc2.', k) for k in layer_keys]
    layer_keys = [re.sub('^head\\.conv', 'box_head.conv', k) for k in
        layer_keys]

    def fpn_map(name):
        """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
        splits = name.split('.')
        norm = '.norm' if 'norm' in splits else ''
        if name.startswith('fpn.inner.'):
            stage = int(splits[2][len('res'):])
            return 'fpn_lateral{}{}.{}'.format(stage, norm, splits[-1])
        elif name.startswith('fpn.res'):
            stage = int(splits[1][len('res'):])
            return 'fpn_output{}{}.{}'.format(stage, norm, splits[-1])
        return name
    layer_keys = [fpn_map(k) for k in layer_keys]
    layer_keys = [k.replace('.[mask].fcn', 'mask_head.mask_fcn') for k in
        layer_keys]
    layer_keys = [re.sub('^\\.mask\\.fcn', 'mask_head.mask_fcn', k) for k in
        layer_keys]
    layer_keys = [k.replace('mask.fcn.logits', 'mask_head.predictor') for k in
        layer_keys]
    layer_keys = [k.replace('conv5.mask', 'mask_head.deconv') for k in
        layer_keys]
    layer_keys = [k.replace('conv.fcn', 'roi_heads.keypoint_head.conv_fcn') for
        k in layer_keys]
    layer_keys = [k.replace('kps.score.lowres',
        'roi_heads.keypoint_head.score_lowres') for k in layer_keys]
    layer_keys = [k.replace('kps.score.', 'roi_heads.keypoint_head.score.') for
        k in layer_keys]
    assert len(set(layer_keys)) == len(layer_keys)
    assert len(original_keys) == len(layer_keys)
    new_weights = {}
    new_keys_to_original_keys = {}
    for orig, renamed in zip(original_keys, layer_keys):
        new_keys_to_original_keys[renamed] = orig
        if renamed.startswith('bbox_pred.') or renamed.startswith(
            'mask_head.predictor.'):
            new_start_idx = 4 if renamed.startswith('bbox_pred.') else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.info(
                'Remove prediction weight for background class in {}. The shape changes from {} to {}.'
                .format(renamed, tuple(weights[orig].shape), tuple(
                new_weights[renamed].shape)))
        elif renamed.startswith('cls_score.'):
            logger.info(
                'Move classification weights for background class in {} from index 0 to index {}.'
                .format(renamed, weights[orig].shape[0] - 1))
            new_weights[renamed] = torch.cat([weights[orig][1:], weights[
                orig][:1]])
        else:
            new_weights[renamed] = weights[orig]
    return new_weights, new_keys_to_original_keys
