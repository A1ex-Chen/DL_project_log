def push_to_hf_dataset():
    all_csvs = sorted(glob.glob(f'{BASE_PATH}/*.csv'))
    collate_csv(all_csvs, FINAL_CSV_FILE)
    csv_path = has_previous_benchmark()
    if csv_path is not None:
        current_results = pd.read_csv(FINAL_CSV_FILE)
        previous_results = pd.read_csv(csv_path)
        numeric_columns = current_results.select_dtypes(include=['float64',
            'int64']).columns
        numeric_columns = [c for c in numeric_columns if c not in [
            'batch_size', 'num_inference_steps', 'actual_gpu_memory (gbs)']]
        for column in numeric_columns:
            previous_results[column] = previous_results[column].map(lambda
                x: filter_float(x))
            current_results[column] = current_results[column].astype(float)
            previous_results[column] = previous_results[column].astype(float)
            percent_change = (current_results[column] - previous_results[
                column]) / previous_results[column] * 100
            current_results[column] = current_results[column].map(str
                ) + percent_change.map(lambda x:
                f" ({'+' if x > 0 else ''}{x:.2f}%)")
            current_results[column] = current_results[column].map(lambda x:
                x.replace(' (nan%)', ''))
        current_results.to_csv(FINAL_CSV_FILE, index=False)
    commit_message = (f'upload from sha: {GITHUB_SHA}' if GITHUB_SHA is not
        None else 'upload benchmark results')
    upload_file(repo_id=REPO_ID, path_in_repo=FINAL_CSV_FILE,
        path_or_fileobj=FINAL_CSV_FILE, repo_type='dataset', commit_message
        =commit_message)
