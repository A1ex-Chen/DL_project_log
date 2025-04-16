@classmethod
def from_args(cls, args, options=''):
    """
        Initialize this class from some cli arguments. Used in train, test.
        """
    for opt in options:
        args.add_argument(*opt.flags, default=None, type=opt.type)
    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.resume is not None:
        resume = Path(args.resume)
        cfg_fname = resume.parent / 'config.json'
    else:
        msg_no_cfg = (
            "Configuration file need to be specified. Add '-c config.json', for example."
            )
        assert args.config is not None, msg_no_cfg
        resume = None
        cfg_fname = Path(args.config)
    config = read_json(cfg_fname)
    if args.config and resume:
        config.update(read_json(args.config))
    modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for
        opt in options}
    return cls(config, resume, modification)
