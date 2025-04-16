def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('result_path', type=str)
    args = parser.parse_args()
    return args
