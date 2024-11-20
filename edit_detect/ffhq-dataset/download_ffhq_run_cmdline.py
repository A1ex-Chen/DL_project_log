def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=
        'Download Flickr-Face-HQ (FFHQ) dataset to current working directory.')
    parser.add_argument('-j', '--json', help=
        'download metadata as JSON (254 MB)', dest='tasks', action=
        'append_const', const='json')
    parser.add_argument('-s', '--stats', help=
        'print statistics about the dataset', dest='tasks', action=
        'append_const', const='stats')
    parser.add_argument('-i', '--images', help=
        'download 1024x1024 images as PNG (89.1 GB)', dest='tasks', action=
        'append_const', const='images')
    parser.add_argument('-t', '--thumbs', help=
        'download 128x128 thumbnails as PNG (1.95 GB)', dest='tasks',
        action='append_const', const='thumbs')
    parser.add_argument('-w', '--wilds', help=
        'download in-the-wild images as PNG (955 GB)', dest='tasks', action
        ='append_const', const='wilds')
    parser.add_argument('-r', '--tfrecords', help=
        'download multi-resolution TFRecords (273 GB)', dest='tasks',
        action='append_const', const='tfrecords')
    parser.add_argument('-a', '--align', help=
        'recreate 1024x1024 images from in-the-wild images', dest='tasks',
        action='append_const', const='align')
    parser.add_argument('--num_threads', help=
        'number of concurrent download threads (default: 32)', type=int,
        default=32, metavar='NUM')
    parser.add_argument('--status_delay', help=
        'time between download status prints (default: 0.2)', type=float,
        default=0.2, metavar='SEC')
    parser.add_argument('--timing_window', help=
        'samples for estimating download eta (default: 50)', type=int,
        default=50, metavar='LEN')
    parser.add_argument('--chunk_size', help=
        'chunk size for each download thread (default: 128)', type=int,
        default=128, metavar='KB')
    parser.add_argument('--num_attempts', help=
        'number of download attempts per file (default: 10)', type=int,
        default=10, metavar='NUM')
    parser.add_argument('--random-shift', help=
        'standard deviation of random crop rectangle jitter', type=float,
        default=0.0, metavar='SHIFT')
    parser.add_argument('--retry-crops', help=
        'retry random shift if crop rectangle falls outside image (up to 1000 times)'
        , dest='retry_crops', default=False, action='store_true')
    parser.add_argument('--no-rotation', help=
        'keep the original orientation of images', dest='no_rotation',
        default=False, action='store_true')
    parser.add_argument('--no-padding', help=
        'do not apply blur-padding outside and near the image borders',
        dest='no_padding', default=False, action='store_true')
    parser.add_argument('--source-dir', help=
        'where to find already downloaded FFHQ source data', default='',
        metavar='DIR')
    args = parser.parse_args()
    if not args.tasks:
        print('No tasks specified. Please see "-h" for help.')
        exit(1)
    run(**vars(args))
