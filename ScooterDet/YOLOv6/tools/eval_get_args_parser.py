def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating',
        add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml',
        help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=
        './weights/yolov6s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size'
        )
    parser.add_argument('--img-size', type=int, default=640, help=
        'inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.03, help=
        'confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help=
        'NMS IoU threshold')
    parser.add_argument('--task', default='val', help='val, test, or speed')
    parser.add_argument('--device', default='0', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help=
        'whether to use fp16 infer')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help=
        'evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help=
        'save evaluation results to save_dir/name')
    parser.add_argument('--shrink_size', type=int, default=0, help=
        'load img resize when test')
    parser.add_argument('--infer_on_rect', default=True, type=
        boolean_string, help=
        'default to run with rectangle image to boost speed.')
    parser.add_argument('--reproduce_640_eval', default=False, action=
        'store_true', help=
        'whether to reproduce 640 infer result, overwrite some config')
    parser.add_argument('--eval_config_file', type=str, default=
        './configs/experiment/eval_640_repro.py', help=
        'config file for repro 640 infer result')
    parser.add_argument('--do_coco_metric', default=True, type=
        boolean_string, help=
        'whether to use pycocotool to metric, set False to close')
    parser.add_argument('--do_pr_metric', default=False, type=
        boolean_string, help=
        'whether to calculate precision, recall and F1, n, set False to close')
    parser.add_argument('--plot_curve', default=True, type=boolean_string,
        help=
        'whether to save plots in savedir when do pr metric, set False to close'
        )
    parser.add_argument('--plot_confusion_matrix', default=False, action=
        'store_true', help=
        'whether to save confusion matrix plots when do pr metric, might cause no harm warning print'
        )
    parser.add_argument('--verbose', default=False, action='store_true',
        help='whether to print metric on each class')
    parser.add_argument('--config-file', default='', type=str, help=
        'experiments description file, lower priority than reproduce_640_eval')
    parser.add_argument('--specific-shape', action='store_true', help=
        'rectangular training')
    parser.add_argument('--height', type=int, default=None, help=
        'image height of model input')
    parser.add_argument('--width', type=int, default=None, help=
        'image width of model input')
    args = parser.parse_args()
    if args.config_file:
        assert os.path.exists(args.config_file), print(
            'Config file {} does not exist'.format(args.config_file))
        cfg = Config.fromfile(args.config_file)
        if not hasattr(cfg, 'eval_params'):
            LOGGER.info("Config file doesn't has eval params config.")
        else:
            eval_params = cfg.eval_params
            for key, value in eval_params.items():
                if key not in args.__dict__:
                    LOGGER.info(f'Unrecognized config {key}, continue')
                    continue
                if isinstance(value, list):
                    if value[1] is not None:
                        args.__dict__[key] = value[1]
                elif value is not None:
                    args.__dict__[key] = value
    if args.reproduce_640_eval:
        assert os.path.exists(args.eval_config_file), print(
            'Reproduce config file {} does not exist'.format(args.
            eval_config_file))
        eval_params = Config.fromfile(args.eval_config_file).eval_params
        eval_model_name = os.path.splitext(os.path.basename(args.weights))[0]
        if eval_model_name not in eval_params:
            eval_model_name = 'default'
        args.shrink_size = eval_params[eval_model_name]['shrink_size']
        args.infer_on_rect = eval_params[eval_model_name]['infer_on_rect']
        args.conf_thres = 0.03
        args.iou_thres = 0.65
        args.task = 'val'
        args.do_coco_metric = True
    LOGGER.info(args)
    return args
