@pytest.fixture
def bitnet_model():
    """A fixture to create an instance of BitNetTransformer model"""
    return BitNetTransformer(num_tokens=20000, dim=512, depth=6, dim_head=
        64, heads=8, ff_mult=4)
