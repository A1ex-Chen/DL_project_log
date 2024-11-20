def get_args_parser():
    parser = argparse.ArgumentParser(description=
        'ULIP training and evaluation', add_help=False)
    parser.add_argument('--output_dir', default='./outputs', type=str, help
        ='output dir')
    parser.add_argument('--pretrain_dataset_name', default='scanrefer',
        type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='scanrefer',
        type=str)
    parser.add_argument('--validate_dataset_name', default=
        'scanrefer_valid', type=str)
    parser.add_argument('--validate_dataset_prompt', default='scanrefer',
        type=str)
    parser.add_argument('--use_height', action='store_true', help=
        'whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help=
        'number of points used for pre-train and test.')
    parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help=
        'number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--lr-start', default=1e-06, type=float, help=
        'initial warmup lr')
    parser.add_argument('--lr_end', default=1e-05, type=float, help=
        'minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help=
        'optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-08, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help=
        'disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help=
        'path to resume from')
    parser.add_argument('--print-freq', default=10, type=int, help=
        'print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar=
        'N', help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help=
        'eval 3d only')
    parser.add_argument('--world-size', default=1, type=int, help=
        'number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help=
        'node rank for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str, help=
        'url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help=
        'Enable WandB logging')
    parser.add_argument('--test_ckpt_addr', default='', help=
        'the ckpt to test 3d zero shot')
    parser.add_argument('--use_scanrefer', action='store_true', default=
        False, help='use pretrain model to train scanrefer')
    parser.add_argument('--use_memory_bank', action='store_true', default=
        True, help='use memory bank to provide negative pair')
    parser.add_argument('--memory_bank_size', type=int, default=1000)
    return parser
