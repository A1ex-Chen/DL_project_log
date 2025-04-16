def _mapping_for_instances(instances):
    mapping = np.zeros(len(instances), dtype='uint8')
    for inst in instances:
        if inst.instance_type == sn.Instance.BACKGROUND:
            continue
        mapping[inst.instance_id] = WNID_TO_NYU[inst.semantic_wordnet_id]
    return mapping
