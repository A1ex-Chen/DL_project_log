def test_one_request_upon_cached(self):
    if torch_device == 'mps':
        return
    use_safetensors = False
    with tempfile.TemporaryDirectory() as tmpdirname:
        with requests_mock.mock(real_http=True) as m:
            UNet2DConditionModel.from_pretrained(
                'hf-internal-testing/tiny-stable-diffusion-torch',
                subfolder='unet', cache_dir=tmpdirname, use_safetensors=
                use_safetensors)
        download_requests = [r.method for r in m.request_history]
        assert download_requests.count('HEAD'
            ) == 2, '2 HEAD requests one for config, one for model'
        assert download_requests.count('GET'
            ) == 2, '2 GET requests one for config, one for model'
        with requests_mock.mock(real_http=True) as m:
            UNet2DConditionModel.from_pretrained(
                'hf-internal-testing/tiny-stable-diffusion-torch',
                subfolder='unet', cache_dir=tmpdirname, use_safetensors=
                use_safetensors)
        cache_requests = [r.method for r in m.request_history]
        assert 'HEAD' == cache_requests[0] and len(cache_requests
            ) == 1, 'We should call only `model_info` to check for _commit hash and `send_telemetry`'
