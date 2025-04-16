def test_exceptions_on_wrong_dtype():
    block = ParallelTransformerBlock(512, 64, 8, 4)
    with pytest.raises(Exception):
        block(torch.randn(32, 512).int())
