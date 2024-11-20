def update_test_map_with_core_pipelines(json_output_file: str):
    print(
        f'\n### ADD CORE PIPELINE TESTS ###\n{_print_list(IMPORTANT_PIPELINES)}'
        )
    with open(json_output_file, 'rb') as fp:
        test_map = json.load(fp)
    test_map['core_pipelines'] = ' '.join(sorted([str(PATH_TO_TESTS /
        f'pipelines/{pipe}') for pipe in IMPORTANT_PIPELINES]))
    if 'pipelines' not in test_map:
        with open(json_output_file, 'w', encoding='UTF-8') as fp:
            json.dump(test_map, fp, ensure_ascii=False)
    pipeline_tests = test_map.pop('pipelines')
    pipeline_tests = pipeline_tests.split(' ')
    updated_pipeline_tests = []
    for pipe in pipeline_tests:
        if pipe == 'tests/pipelines' or Path(pipe).parts[2
            ] in IMPORTANT_PIPELINES:
            continue
        updated_pipeline_tests.append(pipe)
    if len(updated_pipeline_tests) > 0:
        test_map['pipelines'] = ' '.join(sorted(updated_pipeline_tests))
    with open(json_output_file, 'w', encoding='UTF-8') as fp:
        json.dump(test_map, fp, ensure_ascii=False)
