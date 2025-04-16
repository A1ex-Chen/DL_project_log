def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT /
        'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT /
        'yolov3-tiny.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size'
        )
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default
        =640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help=
        'confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help=
        'NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help=
        'maximum detections per image')
    parser.add_argument('--task', default='val', help=
        'train, val, test, speed or study')
    parser.add_argument('--device', default='', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help=
        'max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help=
        'treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help=
        'augmented inference')
    parser.add_argument('--verbose', action='store_true', help=
        'report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help=
        'save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help=
        'save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help=
        'save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help=
        'save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help=
        'save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help=
        'existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help=
        'use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help=
        'use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt
