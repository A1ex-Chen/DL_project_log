def get_model_args(model_args):
    """ the arguments initialize_model will receive """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='resnet50', type=str, required=
        True, help='Network to deploy')
    parser.add_argument('--checkpoint', default=None, type=str, help=
        'The checkpoint of the model. ')
    parser.add_argument('--batch_size', default=1000, type=int, help=
        'Batch size for inference')
    parser.add_argument('--fp16', default=False, action='store_true', help=
        'FP16 inference')
    parser.add_argument('--dump_perf_data', type=str, default=None, help=
        'Directory to dump perf data sample for testing')
    return parser.parse_args(model_args)
