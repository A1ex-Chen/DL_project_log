def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--class_prompt', help=
        'text prompt to retrieve images', required=True, type=str)
    parser.add_argument('--class_data_dir', help='path to save images',
        required=True, type=str)
    parser.add_argument('--num_class_images', help=
        'number of images to download', default=200, type=int)
    return parser.parse_args()
