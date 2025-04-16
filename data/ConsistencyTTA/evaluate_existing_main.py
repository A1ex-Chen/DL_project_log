def main():
    args = parse_args()
    seed_all(args.seed)
    all_outputs = []
    for filename in os.listdir(args.gen_files_path):
        if filename.endswith('.wav'):
            wav_tuple = sf.read(os.path.join(args.gen_files_path, filename))
            all_outputs += [wav_tuple[0]]
    evaluator = EvaluationHelper(sampling_rate=16000, device=device)
    result = evaluator.main(dataset_json_path=args.dataset_json_path,
        generated_files_path=args.gen_files_path, groundtruth_path=args.
        test_references, target_length=args.target_length)
    result['Test Instances'] = len(all_outputs)
    print(result)
