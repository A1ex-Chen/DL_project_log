def fetch_pipeline_objects():
    models = api.list_models(filter=filter)
    downloads = defaultdict(int)
    for model in models:
        is_counted = False
        for tag in model.tags:
            if tag.startswith('diffusers:'):
                is_counted = True
                downloads[tag[len('diffusers:'):]] += model.downloads
        if not is_counted:
            downloads['other'] += model.downloads
    downloads = {k: v for k, v in downloads.items() if v > 0}
    pipeline_objects = filter_pipelines(downloads, PIPELINE_USAGE_CUTOFF)
    return pipeline_objects
