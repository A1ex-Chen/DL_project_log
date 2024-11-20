def process_arguments():
    """ Collect the input arguments according to the syntax
        Return a parser with the arguments
    """
    parser = argparse.ArgumentParser(description=
        'Uses a trained network to predict the input image - flower - name')
    parser.add_argument('--image', action='store', dest='input_image_path',
        default=
        'D:\\Desktop\\部分识别样本\\部分识别样本\\flickr_scraper\\images\\Erthesina_fullo\\4002760475_4d61a3b4a8_o.jpg'
        , help='File path to the input flower image')
    parser.add_argument('--checkpoint', action='store', dest=
        'checkpoint_file_path', default=
        'checkpoint_dir\\resnet50_CUDA5.pth', help=
        'File path to the checkpoint file to use')
    parser.add_argument('--top_k', action='store', dest='topk', default=2,
        type=int, help='top K most likely classes to return')
    parser.add_argument('--mapping', action='store', dest='cat_name_file',
        default='cat_to_name.json', help=
        'file for mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help=
        'Use GPU. The default is CPU')
    return parser.parse_args()
