def __init__(self, dataset_config, ap_iou_thresh=[0.25, 0.5],
    class2type_map=None, exact_eval=True, ap_config_dict=None):
    """
        Args:
            ap_iou_thresh: List of float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
    self.ap_iou_thresh = ap_iou_thresh
    if ap_config_dict is None:
        ap_config_dict = get_ap_config_dict(dataset_config=dataset_config,
            remove_empty_box=exact_eval)
    self.ap_config_dict = ap_config_dict
    self.class2type_map = class2type_map
    self.reset()
