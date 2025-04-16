def ckpt_parser(parser):
    parser.add_argument('--ckpt_restart_mode', type=str, default='auto',
        choices=['off', 'auto', 'required'], help=
        'Mode to restart from a saved checkpoint file, ' +
        "choices are 'off', 'auto', 'required'")
    parser.add_argument('--ckpt_checksum', type=str2bool, default=False,
        help='Checksum the restart file after read+write')
    parser.add_argument('--ckpt_skip_epochs', type=int, default=0, help=
        'Number of epochs to skip before saving epochs')
    parser.add_argument('--ckpt_directory', type=str, default='./save',
        help='Base directory in which to save checkpoints')
    parser.add_argument('--ckpt_save_best', type=str2bool, default=True,
        help='Toggle saving best model')
    parser.add_argument('--ckpt_save_best_metric', type=str, default=
        'val_loss', help='Metric for determining when to save best model')
    parser.add_argument('--ckpt_save_weights_only', type=str2bool, default=
        False, help='Toggle saving only weights (not optimizer) (NYI)')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help=
        'Epoch interval to save checkpoints.  ' +
        'Set to 0 to disable writing checkpoints')
    parser.add_argument('--ckpt_keep_mode', choices=['linear',
        'exponential'], help='Checkpoint saving mode. ' +
        "Choices are 'linear' or 'exponential' ")
    parser.add_argument('--ckpt_keep_limit', type=int, default=1000000,
        help='Limit checkpoints to keep')
    return parser
