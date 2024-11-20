def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT /
        'yolov5s-cls.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT /
        'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=
        int, default=[224], help='inference size h,w')
    parser.add_argument('--device', default='', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help=
        'save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help=
        'do not save images/videos')
    parser.add_argument('--augment', action='store_true', help=
        'augmented inference')
    parser.add_argument('--visualize', action='store_true', help=
        'visualize features')
    parser.add_argument('--update', action='store_true', help=
        'update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls',
        help='save results to project/name')
    parser.add_argument('--name', default='exp', help=
        'save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help=
        'existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help=
        'use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help=
        'use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help=
        'video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt
