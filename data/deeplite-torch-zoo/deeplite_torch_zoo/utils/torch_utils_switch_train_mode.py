@contextmanager
def switch_train_mode(model, is_training=False):
    is_original_mode_training = model.training
    model.train(is_training)
    try:
        yield
    finally:
        model.train(is_original_mode_training)
