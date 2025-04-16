def assert_frozen(module):
    for p in module.parameters():
        assert not p.requires_grad
