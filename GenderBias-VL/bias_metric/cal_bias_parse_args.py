def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--inferencer_type', type=str, required=True)
    parser.add_argument('--base_result', type=str, required=True)
    parser.add_argument('--cf_result', type=str, required=True)
    parser.add_argument('--base_test_file', type=str, required=True)
    parser.add_argument('--cf_test_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./bias_results')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args
