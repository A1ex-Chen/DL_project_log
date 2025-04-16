def parse_args():
    parser = ArgumentParserRGBDSegmentation(description=
        'Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(
            f'Adapting learning rate to {args.lr} because provided batch size differs from default batch size of 8.'
            )
    return args
