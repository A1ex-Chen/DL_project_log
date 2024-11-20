def validate_bboxes(bboxes: list[float, float, float, float, float, str,
    str], is_gt=False) ->None:
    for bbox in bboxes:
        assert len(bbox
            ) == 7, f'bbox must have 7 elements (xmin, ymin, width, height, confidence, class_id, image_name), but {len(bbox)} elements found.'
        assert isinstance(bbox[0], (int, float)
            ), f'xmin must be int or float, but {type(bbox[0])} found.'
        assert isinstance(bbox[1], (int, float)
            ), f'ymin must be int or float, but {type(bbox[1])} found.'
        assert isinstance(bbox[2], (int, float)
            ), f'width must be int or float, but {type(bbox[2])} found.'
        assert isinstance(bbox[3], (int, float)
            ), f'height must be int or float, but {type(bbox[3])} found.'
        if is_gt:
            assert bbox[4
                ] == 1, f'confidence must be 1 for ground truth bbox, but {bbox[4]} found.'
        else:
            assert isinstance(bbox[4], (int, float)
                ), f'confidence must be int or float, but {type(bbox[4])} found.'
        assert isinstance(bbox[5], str
            ), f'class_id must be str, but {type(bbox[5])} found.'
        assert isinstance(bbox[6], str
            ), f'image_name must be str, but {type(bbox[6])} found.'
