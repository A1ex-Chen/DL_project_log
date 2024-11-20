def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating',
        add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml',
        help='dataset yaml file path.')
    parser.add_argument('--weights', type=str, default='./yolov6s.engine',
        help='tensorrt engine file path.')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size'
        )
    parser.add_argument('--img-size', type=int, default=640, help=
        'inference size (pixels)')
    parser.add_argument('--task', default='val', help='can only be val now.')
    parser.add_argument('--device', default='0', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help=
        'evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help=
        'save evaluation results to save_dir/name')
    args = parser.parse_args()
    LOGGER.info(args)
    return args
