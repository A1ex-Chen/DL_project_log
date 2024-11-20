@staticmethod
def save(checkpoint: 'Checkpoint', path_to_checkpoint: str):
    model = checkpoint.model
    algorithm: Algorithm = model.algorithm
    checkpoint_dict = {'epoch': checkpoint.epoch, 'optimizer': checkpoint.
        optimizer, 'model_state_dict': model.state_dict(), 'num_classes':
        model.num_classes, 'preprocessor': model.preprocessor,
        'class_to_category_dict': model.class_to_category_dict,
        'category_to_class_dict': model.category_to_class_dict,
        'algorithm_class': algorithm.__class__, 'algorithm_params': {
        'pretrained': algorithm.pretrained, 'num_frozen_levels': algorithm.
        num_frozen_levels, 'eval_center_crop_ratio': algorithm.
        eval_center_crop_ratio}}
    torch.save(checkpoint_dict, path_to_checkpoint)
