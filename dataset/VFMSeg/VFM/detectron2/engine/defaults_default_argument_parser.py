def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(epilog=epilog or
        f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
"""
        , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-file', default='', metavar='FILE', help=
        'path to config file')
    parser.add_argument('--resume', action='store_true', help=
        'Whether to attempt to resume from the checkpoint directory. See documentation of `DefaultTrainer.resume_or_load()` for what it means.'
        )
    parser.add_argument('--eval-only', action='store_true', help=
        'perform evaluation only')
    parser.add_argument('--num-gpus', type=int, default=1, help=
        'number of gpus *per machine*')
    parser.add_argument('--num-machines', type=int, default=1, help=
        'total number of machines')
    parser.add_argument('--machine-rank', type=int, default=0, help=
        'the rank of this machine (unique per machine)')
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != 'win32' else
        1) % 2 ** 14
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:{}'.format(
        port), help=
        'initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.'
        )
    parser.add_argument('opts', help=
        """
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """
        .strip(), default=None, nargs=argparse.REMAINDER)
    return parser
