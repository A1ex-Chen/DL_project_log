def test_lock_assetsmanager(capsys, working_dir):
    assets_dir = os.path.join(working_dir, 'assets_dir')
    os.makedirs(assets_dir)
    driver_path = os.path.join(working_dir, 'local_driver')
    os.makedirs(os.path.join(driver_path, 'bucket'))
    mng = StorageProvider(provider='local', bucket=driver_path, prefix='prefix'
        )
    data_path = os.path.join(TEST_DIR, 'assets', 'testdata', 'some_data_folder'
        )
    mng.new(data_path, 'category-test/some-data.ext', '0.0')
    script_path = os.path.join(TEST_DIR, 'assets', 'resources',
        'download_asset.py')
    cmd = [sys.executable, script_path, assets_dir, driver_path,
        'category-test/some-data.ext:0.0']

    def run():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess
            .STDOUT)
        stdout, _ = p.communicate()
        stdout = stdout.decode('utf-8')
        print(stdout)
    threads = []
    for _ in range(2):
        t = threading.Thread(target=run)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    captured = capsys.readouterr()
    assert '__ok_from_cache__' in captured.out
    assert '__ok_not_from_cache__' in captured.out
