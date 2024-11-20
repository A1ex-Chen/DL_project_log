def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_cfg', type=str, required=True)
    parser.add_argument('--recipe_cfg', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='../results')
    parser.add_argument('--time', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample_len', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    return args
