def parse_args() ->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image files')
    parser.add_argument('param', help='NCNN param file')
    parser.add_argument('bin', help='NCNN bin file')
    parser.add_argument('--show', action='store_true', help='Show image result'
        )
    parser.add_argument('--out-dir', default='./output', help=
        'Path to output file')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 
        320], help='Image size of height and width')
    parser.add_argument('--max-stride', type=int, default=64, help=
        'Max stride of yolov6 model')
    args = parser.parse_args()
    assert args.max_stride in (32, 64)
    return args
