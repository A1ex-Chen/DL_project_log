def update_metadata(commit_sha: str):
    """
    Update the metadata for the Diffusers repo in `huggingface/diffusers-metadata`.

    Args:
        commit_sha (`str`): The commit SHA on Diffusers corresponding to this update.
    """
    pipelines_table = get_supported_pipeline_table()
    pipelines_table = pd.DataFrame(pipelines_table)
    pipelines_dataset = Dataset.from_pandas(pipelines_table)
    hub_pipeline_tags_json = hf_hub_download(repo_id=
        'huggingface/diffusers-metadata', filename=PIPELINE_TAG_JSON,
        repo_type='dataset')
    with open(hub_pipeline_tags_json) as f:
        hub_pipeline_tags_json = f.read()
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipelines_dataset.to_json(os.path.join(tmp_dir, PIPELINE_TAG_JSON))
        with open(os.path.join(tmp_dir, PIPELINE_TAG_JSON)) as f:
            pipeline_tags_json = f.read()
        hub_pipeline_tags_equal = hub_pipeline_tags_json == pipeline_tags_json
        if hub_pipeline_tags_equal:
            print('No updates, not pushing the metadata files.')
            return
        if commit_sha is not None:
            commit_message = f"""Update with commit {commit_sha}

See: https://github.com/huggingface/diffusers/commit/{commit_sha}"""
        else:
            commit_message = 'Update'
        upload_folder(repo_id='huggingface/diffusers-metadata', folder_path
            =tmp_dir, repo_type='dataset', commit_message=commit_message)
