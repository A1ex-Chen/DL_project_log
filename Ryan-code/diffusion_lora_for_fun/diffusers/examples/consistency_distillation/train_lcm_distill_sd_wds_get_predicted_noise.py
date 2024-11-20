def get_predicted_noise(model_output, timesteps, sample, prediction_type,
    alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == 'epsilon':
        pred_epsilon = model_output
    elif prediction_type == 'sample':
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == 'v_prediction':
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f'Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction` are supported.'
            )
    return pred_epsilon
