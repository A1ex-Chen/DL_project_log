def assert_unfrozen(module):
    for p in module.parameters():
        assert p.requires_grad
