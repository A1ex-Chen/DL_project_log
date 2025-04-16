def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = 'model_index.json'
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path,
            model_index_filename)
    else:
        model_index = hf_hub_download(repo_id=pretrained_model_name_or_path,
            filename=model_index_filename, revision=revision)
    with open(model_index, 'r') as f:
        scheduler_type = json.load(f)['scheduler'][1]
    return scheduler_type
