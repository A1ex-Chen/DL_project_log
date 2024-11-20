def evaluate_model(x, y, model, df):
    return model.evaluate(x, y, batch_size=256)
