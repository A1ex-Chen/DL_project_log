def ULIP_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=
        point_encoder, vision_model=vision_model, context_length=77,
        vocab_size=49408, transformer_width=512, transformer_heads=8,
        transformer_layers=12, pc_feat_dims=pc_feat_dims)
    if not args.evaluate_3d:
        pretrain_slip_model = torch.load(
            './data/initialize_models/slip_base_100ep.pt', map_location=
            torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''):
            param for param_name, param in pretrain_slip_model_params.items()}
        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue
            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]
            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)
    return model
