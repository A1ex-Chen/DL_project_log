def test_cached_files_are_used_when_no_internet(self):
    response_mock = mock.Mock()
    response_mock.status_code = 500
    response_mock.headers = {}
    response_mock.raise_for_status.side_effect = HTTPError
    response_mock.json.return_value = {}
    orig_pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    orig_comps = {k: v for k, v in orig_pipe.components.items() if hasattr(
        v, 'parameters')}
    with mock.patch('requests.request', return_value=response_mock):
        pipe = DiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch',
            safety_checker=None)
        comps = {k: v for k, v in pipe.components.items() if hasattr(v,
            'parameters')}
    for m1, m2 in zip(orig_comps.values(), comps.values()):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                assert False, 'Parameters not the same!'
