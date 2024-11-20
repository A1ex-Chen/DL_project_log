def top_important_features(model, feature_names, n_top=1000):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
    elif hasattr(model, 'coef_'):
        fi = model.coef_
    else:
        return
    features = [(f, n) for f, n in zip(fi, feature_names)]
    top = sorted(features, key=lambda f: abs(f[0]), reverse=True)[:n_top]
    return top
