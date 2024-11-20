@staticmethod
def save(checkpoint: 'Checkpoint', path_to_checkpoint: str):
    model = checkpoint.model
    algorithm: Algorithm = model.algorithm
    backbone: Backbone = algorithm.backbone
    checkpoint_dict = {'epoch': checkpoint.epoch, 'optimizer': checkpoint.
        optimizer, 'model_state_dict': model.state_dict(), 'num_classes':
        model.num_classes, 'preprocessor': model.preprocessor,
        'class_to_category_dict': model.class_to_category_dict,
        'category_to_class_dict': model.category_to_class_dict,
        'algorithm_class': algorithm.__class__, 'algorithm_params': {
        'backbone_class': backbone.__class__, 'backbone_params': {
        'pretrained': backbone.pretrained, 'num_frozen_levels': backbone.
        num_frozen_levels}, 'anchor_ratios': algorithm.anchor_ratios,
        'anchor_sizes': algorithm.anchor_sizes, 'train_rpn_pre_nms_top_n':
        algorithm.train_rpn_pre_nms_top_n, 'train_rpn_post_nms_top_n':
        algorithm.train_rpn_post_nms_top_n, 'eval_rpn_pre_nms_top_n':
        algorithm.eval_rpn_pre_nms_top_n, 'eval_rpn_post_nms_top_n':
        algorithm.eval_rpn_post_nms_top_n, 'num_anchor_samples_per_batch':
        algorithm.num_anchor_samples_per_batch,
        'num_proposal_samples_per_batch': algorithm.
        num_proposal_samples_per_batch, 'num_detections_per_image':
        algorithm.num_detections_per_image, 'anchor_smooth_l1_loss_beta':
        algorithm.anchor_smooth_l1_loss_beta,
        'proposal_smooth_l1_loss_beta': algorithm.
        proposal_smooth_l1_loss_beta, 'proposal_nms_threshold': algorithm.
        proposal_nms_threshold, 'detection_nms_threshold': algorithm.
        detection_nms_threshold}}
    torch.save(checkpoint_dict, path_to_checkpoint)
