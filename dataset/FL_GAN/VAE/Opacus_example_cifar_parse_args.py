def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DP Training')
    parser.add_argument('--grad_sample_mode', type=str, default='hooks')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help
        ='number of total epochs to run')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size-test', default=256, type=int,
        metavar='N', help=
        'mini-batch size for test dataset (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel'
        )
    parser.add_argument('--batch-size', default=2000, type=int, metavar='N',
        help='approximate bacth size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='SGD momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
        metavar='W', help='SGD weight decay', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar
        ='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action=
        'store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int, help=
        'seed for initializing training. ')
    parser.add_argument('--sigma', type=float, default=1.5, metavar='S',
        help='Noise multiplier (default 1.0)')
    parser.add_argument('-c', '--max-per-sample-grad_norm', type=float,
        default=10.0, metavar='C', help=
        'Clip per-sample gradients to this norm (default 1.0)')
    parser.add_argument('--disable-dp', action='store_true', default=False,
        help='Disable privacy training and just train with vanilla SGD')
    parser.add_argument('--secure-rng', action='store_true', default=False,
        help=
        "Enable Secure RNG to have trustworthy privacy guarantees.Comes at a performance cost. Opacus will emit a warning if secure rng is off,indicating that for production use it's recommender to turn it on."
        )
    parser.add_argument('--delta', type=float, default=1e-05, metavar='D',
        help='Target delta (default: 1e-5)')
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint',
        help='path to save check points')
    parser.add_argument('--data-root', type=str, default='data', help=
        'Where CIFAR10 is/will be stored')
    parser.add_argument('--log-dir', type=str, default=
        '/tmp/stat/tensorboard', help='Where Tensorboard log will be stored')
    parser.add_argument('--optim', type=str, default='SGD', help=
        'Optimizer to use (Adam, RMSprop, SGD)')
    parser.add_argument('--lr-schedule', type=str, choices=['constant',
        'cos'], default='cos')
    parser.add_argument('--device', type=str, default='cpu', help=
        'Device on which to run the code.')
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'Local rank if multi-GPU training, -1 for single GPU training. Will be overriden by the environment variables if running on a Slurm cluster.'
        )
    parser.add_argument('--dist_backend', type=str, default='gloo', help=
        'Choose the backend for torch distributed from: gloo, nccl, mpi')
    parser.add_argument('--clip_per_layer', action='store_true', default=
        False, help=
        'Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.'
        )
    parser.add_argument('--debug', type=int, default=0, help=
        'debug level (default: 0)')
    return parser.parse_args()
