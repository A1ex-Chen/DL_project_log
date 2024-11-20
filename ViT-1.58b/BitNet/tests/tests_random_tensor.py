@pytest.fixture
def random_tensor():
    """A fixture to generate a random tensor"""
    return torch.randn(32, 512)
