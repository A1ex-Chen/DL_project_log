def parse_args():
    parser = argparse.ArgumentParser(description=
        'Inference for text to audio generation task.')
    parser.add_argument('--dataset_json_path', type=str, help=
        'Path to the test dataset json file.', default=
        'data/test_audiocaps_subset.json')
    parser.add_argument('--gen_files_path', required=True, type=str, help=
        'Path to the folder that contains the generated files.')
    parser.add_argument('--test_references', type=str, help=
        'Path to the test dataset json file.', default=
        'dataset/audiocaps_test_references/subset')
    parser.add_argument('--seed', default=0, type=int, help=
        'Random seed. Default to 0.')
    parser.add_argument('--target_length', default=970, type=int, help=
        'Audio truncation length (in centiseconds).')
    return parser.parse_args()
