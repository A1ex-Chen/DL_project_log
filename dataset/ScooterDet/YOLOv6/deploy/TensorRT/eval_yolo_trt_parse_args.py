def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLOv6 TensorRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--imgs_dir', type=str, default=
        '../coco/images/val2017', help=
        'directory of validation dataset images.')
    parser.add_argument('--labels_dir', type=str, default=
        '../coco/labels/val2017', help=
        'directory of validation dataset labels.')
    parser.add_argument('--annotations', type=str, default=
        '../coco/annotations/instances_val2017.json', help=
        'coco format annotations of validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1, help=
        'batch size of evaluation.')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 
        640], help='image size')
    parser.add_argument('--model', '-m', type=str, default=
        './weights/yolov5s.trt', help='trt model path')
    parser.add_argument('--conf_thres', type=float, default=0.03, help=
        'confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help=
        'IOU threshold for NMS')
    parser.add_argument('--class_num', type=int, default=3, help=
        'class list for general datasets that must be specified')
    parser.add_argument('--is_coco', action='store_true', help=
        'whether the validation dataset is coco, default is False.')
    parser.add_argument('--shrink_size', type=int, default=4, help=
        'load img with size (img_size - shrink_size), for better performace.')
    parser.add_argument('--visualize', '-v', action='store_true', default=
        False, help='visualize demo')
    parser.add_argument('--num_imgs_to_visualize', type=int, default=10,
        help='number of images to visualize')
    parser.add_argument('--do_pr_metric', action='store_true', help=
        'use pr_metric to evaluate models')
    parser.add_argument('--plot_curve', type=bool, default=True, help=
        'plot curve for pr_metric')
    parser.add_argument('--plot_confusion_matrix', action='store_true',
        help='plot confusion matrix ')
    parser.add_argument('--verbose', action='store_true', help=
        'report mAP by class')
    parser.add_argument('--save_dir', default='', help='whether use pr_metric')
    parser.add_argument('--is_end2end', action='store_true', help=
        'whether the model is end2end (build with NMS)')
    args = parser.parse_args()
    return args
