def evaluate_autoencoder(y_pred, y_test):
    try:
        mse = mean_squared_error(y_pred, y_test)
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
    except Exception:
        r2 = 0
        mse = 0
        corr = 0
    return {'mse': mse, 'r2_score': r2, 'correlation': corr}
