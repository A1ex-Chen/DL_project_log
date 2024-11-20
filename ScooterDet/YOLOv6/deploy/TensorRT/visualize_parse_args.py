def parse_args():
    """Parse input arguments."""
    desc = 'Visualization of YOLO TRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--imgs-dir', type=str, default='./coco_images/',
        help='directory of to be visualized images ./coco_images/')
    parser.add_argument('--visual-dir', type=str, default='./visual_out',
        help='directory of visualized images ./visual_out')
    parser.add_argument('--batch-size', type=int, default=1, help=
        'batch size for training: default 64')
    parser.add_argument('-c', '--category-num', type=int, default=80, help=
        'number of object categories [80]')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 
        640], help='image size')
    parser.add_argument('-m', '--model', type=str, default=
        './weights/yolov5s-simple.trt', help='trt model path')
    parser.add_argument('--conf-thres', type=float, default=0.03, help=
        'object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help=
        'IOU threshold for NMS')
    parser.add_argument('--shrink_size', type=int, default=6, help=
        'load img with size (img_size - shrink_size), for better performace.')
    args = parser.parse_args()
    return args
