def main():
    test_modules = fetch_pipeline_modules_to_test()
    test_modules.extend(ALWAYS_TEST_PIPELINE_MODULES)
    test_modules = list(set(test_modules))
    print(json.dumps(test_modules))
    save_path = f'{PATH_TO_REPO}/reports'
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/test-pipelines.json', 'w') as f:
        json.dump({'pipeline_test_modules': test_modules}, f)
