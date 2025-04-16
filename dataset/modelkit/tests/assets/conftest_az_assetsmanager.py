@pytest.fixture(scope='function')
def az_assetsmanager(request, working_dir):
    subprocess.call(['docker', 'rm', '-f', 'modelkit-storage-azurite-tests'
        ], stderr=subprocess.DEVNULL)
    azurite_proc = subprocess.Popen(['docker', 'run', '-p', '10002:10002',
        '-p', '10001:10001', '-p', '10000:10000', '--name',
        'modelkit-storage-azurite-tests',
        'mcr.microsoft.com/azure-storage/azurite'])
    yield _start_az_manager(working_dir)
    subprocess.call(['docker', 'stop', 'modelkit-storage-azurite-tests'])
    azurite_proc.wait()
