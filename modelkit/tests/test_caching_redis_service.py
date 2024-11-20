@pytest.fixture()
def redis_service(request):
    if 'JENKINS_CI' in os.environ:
        redis_proc = subprocess.Popen(['redis-server'])

        def finalize():
            redis_proc.terminate()
    else:
        subprocess.Popen(['docker', 'run', '--name', 'redis-tests', '-p',
            '6379:6379', 'redis:5'])

        def finalize():
            subprocess.call(['docker', 'rm', '-f', 'redis-tests'])
    request.addfinalizer(finalize)
    rd = redis.Redis(host='localhost', port=6379)
    for _ in range(30):
        try:
            if rd.ping():
                break
        except redis.ConnectionError:
            time.sleep(1)
    yield
