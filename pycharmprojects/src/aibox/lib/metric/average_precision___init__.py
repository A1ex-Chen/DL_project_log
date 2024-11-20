def __init__(self, image_id_to_pred_bboxes_dict: Dict[str, np.ndarray],
    image_id_to_pred_classes_dict: Dict[str, np.ndarray],
    image_id_to_pred_probs_dict: Dict[str, np.ndarray],
    image_id_to_gt_bboxes_dict: Dict[str, np.ndarray],
    image_id_to_gt_classes_dict: Dict[str, np.ndarray],
    image_id_to_gt_difficulties_dict: Dict[str, np.ndarray], num_classes: int):
    super().__init__()
    assert image_id_to_pred_bboxes_dict.keys(
        ) == image_id_to_pred_classes_dict.keys(
        ) == image_id_to_pred_probs_dict.keys(
        ) == image_id_to_gt_bboxes_dict.keys(
        ) == image_id_to_gt_classes_dict.keys(
        ) == image_id_to_gt_difficulties_dict.keys()
    self.image_ids = list(image_id_to_pred_bboxes_dict.keys())
    self.image_id_to_pred_bboxes_dict = image_id_to_pred_bboxes_dict
    self.image_id_to_pred_classes_dict = image_id_to_pred_classes_dict
    self.image_id_to_pred_probs_dict = image_id_to_pred_probs_dict
    self.image_id_to_gt_bboxes_dict = image_id_to_gt_bboxes_dict
    self.image_id_to_gt_classes_dict = image_id_to_gt_classes_dict
    self.image_id_to_gt_difficulties_dict = image_id_to_gt_difficulties_dict
    unfolded_pred_image_ids = []
    unfolded_pred_bboxes = []
    unfolded_pred_classes = []
    unfolded_pred_probs = []
    unfolded_gt_image_ids = []
    unfolded_gt_bboxes = []
    unfolded_gt_classes = []
    unfolded_gt_difficulties = []
    for image_id in self.image_ids:
        pred_bboxes = image_id_to_pred_bboxes_dict[image_id]
        pred_classes = image_id_to_pred_classes_dict[image_id]
        pred_probs = image_id_to_pred_probs_dict[image_id]
        unfolded_pred_image_ids.extend([image_id] * pred_bboxes.shape[0])
        unfolded_pred_bboxes.append(pred_bboxes)
        unfolded_pred_classes.append(pred_classes)
        unfolded_pred_probs.append(pred_probs)
        gt_bboxes = image_id_to_gt_bboxes_dict[image_id]
        gt_classes = image_id_to_gt_classes_dict[image_id]
        gt_difficulties = image_id_to_gt_difficulties_dict[image_id]
        unfolded_gt_image_ids.extend([image_id] * gt_bboxes.shape[0])
        unfolded_gt_bboxes.append(gt_bboxes)
        unfolded_gt_classes.append(gt_classes)
        unfolded_gt_difficulties.append(gt_difficulties)
    self.unfolded_pred_image_ids: List[str] = unfolded_pred_image_ids
    self.unfolded_pred_bboxes: np.ndarray = np.concatenate(unfolded_pred_bboxes
        , axis=0)
    self.unfolded_pred_classes: np.ndarray = np.concatenate(
        unfolded_pred_classes, axis=0)
    self.unfolded_pred_probs: np.ndarray = np.concatenate(unfolded_pred_probs,
        axis=0)
    self.unfolded_gt_image_ids: List[str] = unfolded_gt_image_ids
    self.unfolded_gt_bboxes: np.ndarray = np.concatenate(unfolded_gt_bboxes,
        axis=0)
    self.unfolded_gt_classes: np.ndarray = np.concatenate(unfolded_gt_classes,
        axis=0)
    self.unfolded_gt_difficulties: np.ndarray = np.concatenate(
        unfolded_gt_difficulties, axis=0)
    self.num_classes = num_classes
