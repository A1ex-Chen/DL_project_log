def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description=
        'YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6s.pt',
        help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help=
        'the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help=
        'whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help=
        'the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help=
        'data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 
        640], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.4, help=
        'confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help=
        'NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help=
        'maximal inferences per image.')
    parser.add_argument('--device', default='0', help=
        'device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help=
        'save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help=
        'do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help=
        'directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help=
        'show inference results')
    parser.add_argument('--classes', nargs='+', type=int, help=
        'filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help=
        'class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference', help=
        'save inference results to project/name.')
    parser.add_argument('--name', default='exp', help=
        'save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true',
        help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true',
        help='hide confidences.')
    parser.add_argument('--half', action='store_true', help=
        'whether to use FP16 half-precision inference.')
    args = parser.parse_args()
    LOGGER.info(args)
    return args
