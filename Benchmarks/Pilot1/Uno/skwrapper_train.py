def train(model, x, y, features=None, classify=False, threads=-1, prefix='',
    name=None, save=False):
    verify_path(prefix)
    model, model_name = get_model(model, threads, classify=classify)
    model.fit(x, y)
    name = name or model_name
    if save:
        model_desc_fname = '{}.{}.description'.format(prefix, name)
        with open(model_desc_fname, 'w') as f:
            f.write('{}\n'.format(model))
        if features:
            top_features = top_important_features(model, features)
            if top_features is not None:
                fea_fname = '{}.{}.features'.format(prefix, name)
                with open(fea_fname, 'w') as fea_file:
                    fea_file.write(sprint_features(top_features))
        model_fname = '{}.{}.model.pkl'.format(prefix, name)
        with open(model_fname, 'wb') as f:
            pickle.dump(model, f)
        return model_fname
