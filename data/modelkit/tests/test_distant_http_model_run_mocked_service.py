@pytest.fixture(scope='module')
def run_mocked_service():
    proc = subprocess.Popen(['uvicorn', 'mocked_service:app'], cwd=os.path.
        join(TEST_DIR, 'testdata'), stdout=subprocess.PIPE, stderr=
        subprocess.PIPE)
    done = False
    for _ in range(300):
        try:
            requests.post('http://localhost:8000/api/path/endpoint', data=
                json.dumps({'ok': 'ok'}))
            done = True
            break
        except requests.ConnectionError:
            time.sleep(0.01)
    if not done:
        _stop_mocked_service_and_print_stderr(proc)
        raise Exception('Could not start mocked service.')
    yield proc
    proc.terminate()
