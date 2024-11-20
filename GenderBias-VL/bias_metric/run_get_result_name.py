def get_result_name(sub_dir):
    candidates = []
    for file in os.listdir(sub_dir):
        file = str(file)
        if file.startswith('OccBias_'):
            candidates.append(file)
    assert len(candidates) == 1 or len(candidates
        ) == 0, f'check in {sub_dir}, candidates: {len(candidates)}'
    if len(candidates) == 0:
        return None
    return os.path.join(sub_dir, candidates[0])
