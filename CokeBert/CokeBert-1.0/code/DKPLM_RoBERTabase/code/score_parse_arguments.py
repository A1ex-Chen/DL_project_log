def parse_arguments(gold_file, pred_file):
    parser = argparse.ArgumentParser(description=
        'Score a prediction file using the gold labels.')
    parser.add_argument('--gold_file', default=gold_file)
    parser.add_argument('--pred_file', default=pred_file)
    args = parser.parse_args()
    return args
