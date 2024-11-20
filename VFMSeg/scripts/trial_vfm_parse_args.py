def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument('--cfg', dest='config_file', default='', metavar=
        'FILE', help='path to config file', type=str)
    parser.add_argument('ckpt2d', type=str, help=
        'path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help=
        'path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help=
        'generate pseudo-labels')
    parser.add_argument('--save-ensemble', action='store_true', help=
        'Whether to save the 2D+3D ensembling pseudo labels.')
    parser.add_argument('--vfmlab', action='store_true', help=
        'Whether to save the VFM generated labels.')
    parser.add_argument('--vfm_pth', type=str, help=
        'the pretrained VFM weights.')
    parser.add_argument('--vfm_cfg', type=str, help=
        'the configuration for VFM.')
    parser.add_argument('opts', help=
        'Modify config options using the command-line', default=None, nargs
        =argparse.REMAINDER)
    args = parser.parse_args()
    return args
