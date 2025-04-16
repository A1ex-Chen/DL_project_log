def get_ap_config_dict(remove_empty_box=True, use_3d_nms=True, nms_iou=0.25,
    use_old_type_nms=False, cls_nms=True, per_class_proposal=True,
    use_cls_confidence_only=False, conf_thresh=0.05, no_nms=False,
    dataset_config=None):
    """
    Default mAP evaluation settings for VoteNet
    """
    config_dict = {'remove_empty_box': remove_empty_box, 'use_3d_nms':
        use_3d_nms, 'nms_iou': nms_iou, 'use_old_type_nms':
        use_old_type_nms, 'cls_nms': cls_nms, 'per_class_proposal':
        per_class_proposal, 'use_cls_confidence_only':
        use_cls_confidence_only, 'conf_thresh': conf_thresh, 'no_nms':
        no_nms, 'dataset_config': dataset_config}
    return config_dict
