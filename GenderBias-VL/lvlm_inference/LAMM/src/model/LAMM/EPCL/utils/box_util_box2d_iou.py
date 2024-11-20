def box2d_iou(box1, box2):
    """Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    """
    return get_iou({'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1
        [3]}, {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]})
