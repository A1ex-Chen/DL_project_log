def get_predicted_original_sample(model_output, timesteps, sample,
    prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == 'epsilon':
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == 'sample':
        pred_x_0 = model_output
    elif prediction_type == 'v_prediction':
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f'Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction` are supported.'
            )
    return pred_x_0
