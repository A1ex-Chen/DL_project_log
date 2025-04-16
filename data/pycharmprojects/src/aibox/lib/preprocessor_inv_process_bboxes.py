@classmethod
def inv_process_bboxes(cls, process_dict: Dict[str, Any], bboxes: Tensor
    ) ->Tensor:
    inv_bboxes = bboxes.clone()
    inv_bboxes[:, [0, 2]] /= process_dict[cls.PROCESS_KEY_WIDTH_SCALE]
    inv_bboxes[:, [1, 3]] /= process_dict[cls.PROCESS_KEY_HEIGHT_SCALE]
    return inv_bboxes
