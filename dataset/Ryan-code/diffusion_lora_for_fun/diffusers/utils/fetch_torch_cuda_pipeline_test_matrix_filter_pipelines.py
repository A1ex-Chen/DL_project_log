def filter_pipelines(usage_dict, usage_cutoff=10000):
    output = []
    for diffusers_object, usage in usage_dict.items():
        if usage < usage_cutoff:
            continue
        is_diffusers_pipeline = hasattr(diffusers.pipelines, diffusers_object)
        if not is_diffusers_pipeline:
            continue
        output.append(diffusers_object)
    return output
