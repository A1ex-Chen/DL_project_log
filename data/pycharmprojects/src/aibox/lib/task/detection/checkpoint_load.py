@staticmethod
def load(path_to_checkpoint: str, device: torch.device) ->'Checkpoint':
    checkpoint_dict = torch.load(path_to_checkpoint, map_location=device)
    num_classes = checkpoint_dict['num_classes']
    backbone_class = checkpoint_dict['algorithm_params']['backbone_class']
    backbone_params = checkpoint_dict['algorithm_params']['backbone_params']
    backbone: Backbone = backbone_class(pretrained=backbone_params[
        'pretrained'], num_frozen_levels=backbone_params['num_frozen_levels'])
    algorithm_class = checkpoint_dict['algorithm_class']
    algorithm_params = checkpoint_dict['algorithm_params']
    algorithm: Algorithm = algorithm_class(num_classes, backbone,
        anchor_ratios=algorithm_params['anchor_ratios'], anchor_sizes=
        algorithm_params['anchor_sizes'], train_rpn_pre_nms_top_n=
        algorithm_params['train_rpn_pre_nms_top_n'],
        train_rpn_post_nms_top_n=algorithm_params[
        'train_rpn_post_nms_top_n'], eval_rpn_pre_nms_top_n=
        algorithm_params['eval_rpn_pre_nms_top_n'], eval_rpn_post_nms_top_n
        =algorithm_params['eval_rpn_post_nms_top_n'],
        num_anchor_samples_per_batch=algorithm_params[
        'num_anchor_samples_per_batch'], num_proposal_samples_per_batch=
        algorithm_params['num_proposal_samples_per_batch'],
        num_detections_per_image=algorithm_params[
        'num_detections_per_image'], anchor_smooth_l1_loss_beta=
        algorithm_params['anchor_smooth_l1_loss_beta'],
        proposal_smooth_l1_loss_beta=algorithm_params[
        'proposal_smooth_l1_loss_beta'], proposal_nms_threshold=
        algorithm_params['proposal_nms_threshold'], detection_nms_threshold
        =algorithm_params['detection_nms_threshold'])
    model = Model(algorithm, num_classes, preprocessor=checkpoint_dict[
        'preprocessor'], class_to_category_dict=checkpoint_dict[
        'class_to_category_dict'], category_to_class_dict=checkpoint_dict[
        'category_to_class_dict'])
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.to(device)
    checkpoint = Checkpoint(epoch=checkpoint_dict['epoch'], model=model,
        optimizer=checkpoint_dict['optimizer'])
    return checkpoint
