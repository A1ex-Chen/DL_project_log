def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100', help=
        'cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--pretraining-dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=300, help=
        'total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help=
        'total batch size for all GPUs')
    parser.add_argument('--test-batch-size', type=int, default=256, help=
        'testing batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default
        =224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help=
        'only save final checkpoint')
    parser.add_argument('--device', default='', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help=
        'max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help=
        'save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help=
        'existing project/name ok, do not increment')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW',
        'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help=
        'initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-05, help=
        'weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help=
        'Label smoothing epsilon')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help=
        'Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--kd_model_name', default=None, type=str)
    parser.add_argument('--kd_model_checkpoint', default=None, type=str)
    parser.add_argument('--alpha_kd', default=5, type=float)
    parser.add_argument('--use_kd_loss_only', action='store_true', default=
        False)
    parser.add_argument('--dryrun', action='store_true', help=
        'Dry run mode for testing')
    return parser.parse_known_args()[0] if known else parser.parse_args()
