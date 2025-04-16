def add_parser_arguments(parser, skip_arch=False):
    parser.add_argument('--data', metavar='DIR', default='/imagenet', help=
        'path to dataset')
    parser.add_argument('--data-backend', metavar='BACKEND', default=
        'pytorch', choices=DATA_BACKEND_CHOICES, help='data backend: ' +
        ' | '.join(DATA_BACKEND_CHOICES) + ' (default: pytorch)')
    parser.add_argument('--interpolation', metavar='INTERPOLATION', default
        ='bilinear', help=
        'interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)'
        )
    if not skip_arch:
        model_names = available_models().keys()
        parser.add_argument('--arch', '-a', metavar='ARCH', default=
            'resnet50', choices=model_names, help='model architecture: ' +
            ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
        help='number of data loading workers (default: 5)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help
        ='number of total epochs to run')
    parser.add_argument('--run-epochs', default=-1, type=int, metavar='N',
        help='run only N epochs, used for checkpointing runs')
    parser.add_argument('--early-stopping-patience', default=-1, type=int,
        metavar='N', help=
        'early stopping after N epochs without validation accuracy improving')
    parser.add_argument('--image-size', default=None, type=int, help=
        'resolution of image')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256) per gpu')
    parser.add_argument('--optimizer-batch-size', default=-1, type=int,
        metavar='N', help=
        'size of a total batch size, for simulating bigger batches using gradient accumulation'
        )
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-schedule', default='step', type=str, metavar=
        'SCHEDULE', choices=['step', 'linear', 'cosine'], help=
        'Type of LR schedule: {}, {}, {}'.format('step', 'linear', 'cosine'))
    parser.add_argument('--end-lr', default=0, type=float)
    parser.add_argument('--warmup', default=0, type=int, metavar='E', help=
        'number of warmup epochs')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
        metavar='S', help='label smoothing')
    parser.add_argument('--mixup', default=0.0, type=float, metavar='ALPHA',
        help='mixup alpha')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=(
        'sgd', 'rmsprop'))
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=
        float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--bn-weight-decay', action='store_true', help=
        'use weight_decay on batch normalization learnable parameters, (default: false)'
        )
    parser.add_argument('--rmsprop-alpha', default=0.9, type=float, help=
        'value of alpha parameter in rmsprop optimizer (default: 0.9)')
    parser.add_argument('--rmsprop-eps', default=0.001, type=float, help=
        'value of eps parameter in rmsprop optimizer (default: 1e-3)')
    parser.add_argument('--nesterov', action='store_true', help=
        'use nesterov momentum, (default: false)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar
        ='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument('--static-loss-scale', type=float, default=1, help=
        'Static loss scale, positive power of 2 values can improve amp convergence.'
        )
    parser.add_argument('--dynamic-loss-scale', action='store_true', help=
        'Use dynamic loss scaling.  If supplied, this argument supersedes ' +
        '--static-loss-scale.')
    parser.add_argument('--prof', type=int, default=-1, metavar='N', help=
        'Run only N iterations')
    parser.add_argument('--amp', action='store_true', help=
        'Run model AMP (automatic mixed precision) mode.')
    parser.add_argument('--seed', default=None, type=int, help=
        'random seed used for numpy and pytorch')
    parser.add_argument('--gather-checkpoints', action='store_true', help=
        'Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored'
        )
    parser.add_argument('--raport-file', default='experiment_raport.json',
        type=str, help='file in which to store JSON experiment raport')
    parser.add_argument('--evaluate', action='store_true', help=
        'evaluate checkpoint/model')
    parser.add_argument('--training-only', action='store_true', help=
        'do not evaluate')
    parser.add_argument('--no-checkpoints', action='store_false', dest=
        'save_checkpoints', help=
        'do not store any checkpoints, useful for benchmarking')
    parser.add_argument('--checkpoint-filename', default=
        'checkpoint.pth.tar', type=str)
    parser.add_argument('--workspace', type=str, default='./', metavar=
        'DIR', help='path to directory where checkpoints will be stored')
    parser.add_argument('--memory-format', type=str, default='nchw',
        choices=['nchw', 'nhwc'], help='memory layout, nchw or nhwc')
    parser.add_argument('--use-ema', default=None, type=float, help='use EMA')
    parser.add_argument('--augmentation', type=str, default=None, choices=[
        None, 'autoaugment'], help='augmentation method')
    parser.add_argument('--num-classes', type=int, default=None, required=
        False, help='number of classes')
