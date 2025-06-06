def _get_lvis_instances_meta_v0_5():
    assert len(LVIS_V0_5_CATEGORIES) == 1230
    cat_ids = [k['id'] for k in LVIS_V0_5_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(cat_ids
        ), 'Category ids are not in [1, #categories], as expected'
    lvis_categories = sorted(LVIS_V0_5_CATEGORIES, key=lambda x: x['id'])
    thing_classes = [k['synonyms'][0] for k in lvis_categories]
    meta = {'thing_classes': thing_classes}
    return meta
