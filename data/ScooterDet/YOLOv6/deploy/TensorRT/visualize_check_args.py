def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.exists(args.visual_dir):
        print('Directory {} does not exist, create it'.format(args.visual_dir))
        os.makedirs(args.visual_dir)
