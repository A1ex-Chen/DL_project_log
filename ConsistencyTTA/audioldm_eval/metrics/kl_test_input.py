def test_input(featuresdict_1, featuresdict_2, feat_layer_name,
    dataset_name, classes):
    assert feat_layer_name == 'logits', 'This KL div metric is implemented on logits.'
    assert 'file_path_' in featuresdict_1 and 'file_path_' in featuresdict_2, 'File paths are missing'
    assert len(featuresdict_1) >= len(featuresdict_2
        ), 'There are more samples in input1, than in input2'
    assert len(featuresdict_1) % len(featuresdict_2
        ) == 0, 'Size of input1 is not a multiple of input1 size.'
    if dataset_name == 'vas':
        assert classes is not None, f'Specify classes if you are using vas dataset. Now `classes` – {classes}'
        print(
            'KL: when FakesFolder on VAS is used as a dataset, we assume the original labels were sorted'
            ,
            'to produce the target_ids. E.g. `baby` -> `cls_0`; `cough` -> `cls_1`; `dog` -> `cls_2`.'
            )
