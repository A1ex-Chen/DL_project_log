def convert_bboxes(bboxes: (pd.DataFrame | BBoxDataFrame | list | tuple)
    ) ->list[float, float, float, float, float, str, str]:
    """Convert bboxes to tuples of (xmin, ymin, width, height, confidence, class_id, image_name).

    Args:
        bboxes (pd.DataFrame | BBoxDataFrame | list | tuple): bboxes to convert.

    Returns:
        list[float, float, float, float, float, str, str]: converted bboxes.
    """
    if isinstance(bboxes, pd.DataFrame) or isinstance(bboxes, BBoxDataFrame):
        bboxes = bboxes.values.tolist()
    elif isinstance(bboxes, list):
        bboxes = [tuple(bbox) for bbox in bboxes]
    for i, bbox in enumerate(bboxes):
        try:
            bboxes[i] = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(
                bbox[3]), float(bbox[4]), str(bbox[5]), str(bbox[6])
        except IndexError:
            raise IndexError(
                f'bbox must have 7 elements (xmin, ymin, width, height, confidence, class_id, image_name), but {len(bbox)} elements found.'
                )
        except ValueError as e:
            expected_types = ('float', 'float', 'float', 'float', 'float',
                'str', 'str')
            actual_types = tuple(type(x).__name__ for x in bbox)
            labels = ('xmin', 'ymin', 'width', 'height', 'confidence',
                'class_id', 'image_name')
            comparison = '\n'.join([f'{label:<10} {expected:<10} {actual}' for
                label, expected, actual in zip(labels, expected_types,
                actual_types)])
            msg = f"""Expected types and actual types don't match:

Label      Expected   Actual
{comparison}

Original error message: {str(e)}"""
            raise ValueError(msg)
    validate_bboxes(bboxes)
    return bboxes
