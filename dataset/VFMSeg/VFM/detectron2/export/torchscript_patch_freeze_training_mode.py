@contextmanager
def freeze_training_mode(model):
    """
    A context manager that annotates the "training" attribute of every submodule
    to constant, so that the training codepath in these modules can be
    meta-compiled away. Upon exiting, the annotations are reverted.
    """
    classes = {type(x) for x in model.modules()}
    classes = {x for x in classes if not hasattr(x, '__constants__')}
    for cls in classes:
        cls.__annotations__['training'] = torch.jit.Final[bool]
    yield
    for cls in classes:
        cls.__annotations__['training'] = bool
