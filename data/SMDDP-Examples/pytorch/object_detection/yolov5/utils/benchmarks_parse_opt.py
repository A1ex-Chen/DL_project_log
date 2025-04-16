def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt',
        help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default
        =640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--data', type=str, default=ROOT /
        'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--device', default='', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help=
        'use FP16 half-precision inference')
    parser.add_argument('--test', action='store_true', help='test exports only'
        )
    parser.add_argument('--pt-only', action='store_true', help=
        'test PyTorch only')
    parser.add_argument('--hard-fail', action='store_true', help=
        'throw error on benchmark failure')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt
