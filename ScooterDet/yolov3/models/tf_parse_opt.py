def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT /
        'yolov3-tiny.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=
        int, default=[640], help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help=
        'dynamic batch size')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt
