def process_targets_for_training(padded_image_size, boxes, classes, params):
    input_anchors = anchors.AnchorGenerator(params['min_level'], params[
        'max_level'], params['num_scales'], params['aspect_ratios'], params
        ['anchor_scale'], padded_image_size)
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params[
        'num_classes'], params['rpn_positive_overlap'], params[
        'rpn_negative_overlap'], params['rpn_batch_size_per_im'], params[
        'rpn_fg_fraction'])
    return anchor_labeler.label_anchors(boxes, classes), input_anchors
