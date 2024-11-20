@staticmethod
def load(path_to_checkpoint: str, device: torch.device) ->'Checkpoint':
    checkpoint_dict = torch.load(path_to_checkpoint, map_location=device)
    num_classes = checkpoint_dict['num_classes']
    algorithm_class = checkpoint_dict['algorithm_class']
    algorithm_params = checkpoint_dict['algorithm_params']
    algorithm: Algorithm = algorithm_class(num_classes, image_min_side=
        algorithm_params['image_min_side'], image_max_side=algorithm_params
        ['image_max_side'])
    model = Model(algorithm, num_classes, preprocessor=checkpoint_dict[
        'preprocessor'], class_to_category_dict=checkpoint_dict[
        'class_to_category_dict'], category_to_class_dict=checkpoint_dict[
        'category_to_class_dict'])
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.to(device)
    checkpoint = Checkpoint(epoch=checkpoint_dict['epoch'], model=model,
        optimizer=checkpoint_dict['optimizer'])
    return checkpoint
