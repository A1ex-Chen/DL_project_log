def has_previous_benchmark() ->str:
    csv_path = None
    try:
        csv_path = hf_hub_download(repo_id=REPO_ID, repo_type='dataset',
            filename=FINAL_CSV_FILE)
    except EntryNotFoundError:
        csv_path = None
    return csv_path
