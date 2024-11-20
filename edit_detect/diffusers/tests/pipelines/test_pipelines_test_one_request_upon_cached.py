def test_one_request_upon_cached(self):
    if torch_device == 'mps':
        return
    with tempfile.TemporaryDirectory() as tmpdirname:
        with requests_mock.mock(real_http=True) as m:
            DiffusionPipeline.download(
                'hf-internal-testing/tiny-stable-diffusion-pipe', cache_dir
                =tmpdirname)
        download_requests = [r.method for r in m.request_history]
        assert download_requests.count('HEAD') == 15, '15 calls to files'
        assert download_requests.count('GET'
            ) == 17, '15 calls to files + model_info + model_index.json'
        assert len(download_requests
            ) == 32, '2 calls per file (15 files) + send_telemetry, model_info and model_index.json'
        with requests_mock.mock(real_http=True) as m:
            DiffusionPipeline.download(
                'hf-internal-testing/tiny-stable-diffusion-pipe',
                safety_checker=None, cache_dir=tmpdirname)
        cache_requests = [r.method for r in m.request_history]
        assert cache_requests.count('HEAD'
            ) == 1, 'model_index.json is only HEAD'
        assert cache_requests.count('GET') == 1, 'model info is only GET'
        assert len(cache_requests
            ) == 2, 'We should call only `model_info` to check for _commit hash and `send_telemetry`'
