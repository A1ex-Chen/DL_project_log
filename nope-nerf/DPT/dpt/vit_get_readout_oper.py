def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == 'project':
        readout_oper = [ProjectReadout(vit_features, start_index) for
            out_feat in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper
