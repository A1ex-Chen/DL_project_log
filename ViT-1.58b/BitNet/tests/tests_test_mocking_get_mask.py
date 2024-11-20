def test_mocking_get_mask(monkeypatch, random_tensor):
    mock_mask = torch.zeros(100, 100)
    monkeypatch.setattr(ParallelTransformerBlock, 'get_mask', lambda self,
        n, device: mock_mask)
    block = ParallelTransformerBlock(512, 64, 8, 4)
    assert torch.equal(block.get_mask(100, random_tensor.device), mock_mask)
