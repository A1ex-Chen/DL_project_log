def test_cached_files_are_used_when_no_internet(self):
    response_mock = mock.Mock()
    response_mock.status_code = 500
    response_mock.headers = {}
    response_mock.raise_for_status.side_effect = HTTPError
    response_mock.json.return_value = {}
    orig_model = UNet2DConditionModel.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', subfolder='unet')
    with mock.patch('requests.request', return_value=response_mock):
        model = UNet2DConditionModel.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'unet', local_files_only=True)
    for p1, p2 in zip(orig_model.parameters(), model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            assert False, 'Parameters not the same!'
