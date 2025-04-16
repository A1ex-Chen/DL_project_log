@staticmethod
def c2_preprocess(box_lists):
    assert all(isinstance(x, Boxes) for x in box_lists)
    if all(isinstance(x, Caffe2Boxes) for x in box_lists):
        assert len(box_lists) == 1
        pooler_fmt_boxes = box_lists[0].tensor
    else:
        pooler_fmt_boxes = poolers.convert_boxes_to_pooler_format(box_lists)
    return pooler_fmt_boxes
