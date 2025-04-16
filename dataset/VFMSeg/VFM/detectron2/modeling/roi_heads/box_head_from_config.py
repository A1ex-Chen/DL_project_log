@classmethod
def from_config(cls, cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    return {'input_shape': input_shape, 'conv_dims': [conv_dim] * num_conv,
        'fc_dims': [fc_dim] * num_fc, 'conv_norm': cfg.MODEL.ROI_BOX_HEAD.NORM}
