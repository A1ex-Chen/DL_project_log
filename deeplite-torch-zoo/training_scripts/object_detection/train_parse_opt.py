def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolo5s', help=
        'zoo model name')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretraining_dataset', type=str, default='coco')
    parser.add_argument('--data_root', type=str, default=None, help=
        'dataset path')
    parser.add_argument('--dataset', type=str, default='coco128', help=
        'dataset name')
    parser.add_argument('--hyp', type=str, default=ROOT /
        'hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help=
        'total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help=
        'total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default
        =640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help=
        'rectangular training')
    parser.add_argument('--nosave', action='store_true', help=
        'only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help=
        'only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help=
        'disable AutoAnchor')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help=
        'image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help=
        'use weighted image selection for training')
    parser.add_argument('--device', default='', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help=
        'vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help=
        'train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam',
        'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help=
        'use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help=
        'max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help=
        'save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help=
        'existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help=
        'cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help=
        'Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help=
        'EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help=
        'Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help=
        'Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help=
        'Global training seed')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--dryrun', action='store_true', help=
        'Dry run mode for testing')
    return parser.parse_known_args()[0] if known else parser.parse_args()
