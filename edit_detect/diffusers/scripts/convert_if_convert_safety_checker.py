def convert_safety_checker(p_head_path, w_head_path):
    state_dict = {}
    p_head = np.load(p_head_path)
    p_head_weights = p_head['weights']
    p_head_weights = torch.from_numpy(p_head_weights)
    p_head_weights = p_head_weights.unsqueeze(0)
    p_head_biases = p_head['biases']
    p_head_biases = torch.from_numpy(p_head_biases)
    p_head_biases = p_head_biases.unsqueeze(0)
    state_dict['p_head.weight'] = p_head_weights
    state_dict['p_head.bias'] = p_head_biases
    w_head = np.load(w_head_path)
    w_head_weights = w_head['weights']
    w_head_weights = torch.from_numpy(w_head_weights)
    w_head_weights = w_head_weights.unsqueeze(0)
    w_head_biases = w_head['biases']
    w_head_biases = torch.from_numpy(w_head_biases)
    w_head_biases = w_head_biases.unsqueeze(0)
    state_dict['w_head.weight'] = w_head_weights
    state_dict['w_head.bias'] = w_head_biases
    vision_model = CLIPVisionModelWithProjection.from_pretrained(
        'openai/clip-vit-large-patch14')
    vision_model_state_dict = vision_model.state_dict()
    for key, value in vision_model_state_dict.items():
        key = f'vision_model.{key}'
        state_dict[key] = value
    config = CLIPConfig.from_pretrained('openai/clip-vit-large-patch14')
    safety_checker = IFSafetyChecker(config)
    safety_checker.load_state_dict(state_dict)
    return safety_checker
