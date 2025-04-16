@pytest.fixture(scope='function')
def s3_assetsmanager(request, working_dir):
    subprocess.call(['docker', 'rm', '-f', 'modelkit-storage-minio-tests'],
        stderr=subprocess.DEVNULL)
    minio_proc = subprocess.Popen(['docker', 'run', '-p', '9000:9000',
        '--name', 'modelkit-storage-minio-tests', 'minio/minio', 'server',
        '/data'])
    yield _start_s3_manager(working_dir)
    subprocess.call(['docker', 'stop', 'modelkit-storage-minio-tests'])
    minio_proc.wait()
