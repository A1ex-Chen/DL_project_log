def convert_adapter(src_state, in_channels):
    original_body_length = max([int(x.split('.')[1]) for x in src_state.
        keys() if 'body.' in x]) + 1
    assert original_body_length == 8
    assert src_state['body.0.block1.weight'].shape == (320, 320, 3, 3)
    assert src_state['body.2.in_conv.weight'].shape == (640, 320, 1, 1)
    assert src_state['body.4.in_conv.weight'].shape == (1280, 640, 1, 1)
    assert src_state['body.6.block1.weight'].shape == (1280, 1280, 3, 3)
    res_state = {'adapter.conv_in.weight': src_state.pop('conv_in.weight'),
        'adapter.conv_in.bias': src_state.pop('conv_in.bias'),
        'adapter.body.0.resnets.0.block1.weight': src_state.pop(
        'body.0.block1.weight'), 'adapter.body.0.resnets.0.block1.bias':
        src_state.pop('body.0.block1.bias'),
        'adapter.body.0.resnets.0.block2.weight': src_state.pop(
        'body.0.block2.weight'), 'adapter.body.0.resnets.0.block2.bias':
        src_state.pop('body.0.block2.bias'),
        'adapter.body.0.resnets.1.block1.weight': src_state.pop(
        'body.1.block1.weight'), 'adapter.body.0.resnets.1.block1.bias':
        src_state.pop('body.1.block1.bias'),
        'adapter.body.0.resnets.1.block2.weight': src_state.pop(
        'body.1.block2.weight'), 'adapter.body.0.resnets.1.block2.bias':
        src_state.pop('body.1.block2.bias'),
        'adapter.body.1.in_conv.weight': src_state.pop(
        'body.2.in_conv.weight'), 'adapter.body.1.in_conv.bias': src_state.
        pop('body.2.in_conv.bias'),
        'adapter.body.1.resnets.0.block1.weight': src_state.pop(
        'body.2.block1.weight'), 'adapter.body.1.resnets.0.block1.bias':
        src_state.pop('body.2.block1.bias'),
        'adapter.body.1.resnets.0.block2.weight': src_state.pop(
        'body.2.block2.weight'), 'adapter.body.1.resnets.0.block2.bias':
        src_state.pop('body.2.block2.bias'),
        'adapter.body.1.resnets.1.block1.weight': src_state.pop(
        'body.3.block1.weight'), 'adapter.body.1.resnets.1.block1.bias':
        src_state.pop('body.3.block1.bias'),
        'adapter.body.1.resnets.1.block2.weight': src_state.pop(
        'body.3.block2.weight'), 'adapter.body.1.resnets.1.block2.bias':
        src_state.pop('body.3.block2.bias'),
        'adapter.body.2.in_conv.weight': src_state.pop(
        'body.4.in_conv.weight'), 'adapter.body.2.in_conv.bias': src_state.
        pop('body.4.in_conv.bias'),
        'adapter.body.2.resnets.0.block1.weight': src_state.pop(
        'body.4.block1.weight'), 'adapter.body.2.resnets.0.block1.bias':
        src_state.pop('body.4.block1.bias'),
        'adapter.body.2.resnets.0.block2.weight': src_state.pop(
        'body.4.block2.weight'), 'adapter.body.2.resnets.0.block2.bias':
        src_state.pop('body.4.block2.bias'),
        'adapter.body.2.resnets.1.block1.weight': src_state.pop(
        'body.5.block1.weight'), 'adapter.body.2.resnets.1.block1.bias':
        src_state.pop('body.5.block1.bias'),
        'adapter.body.2.resnets.1.block2.weight': src_state.pop(
        'body.5.block2.weight'), 'adapter.body.2.resnets.1.block2.bias':
        src_state.pop('body.5.block2.bias'),
        'adapter.body.3.resnets.0.block1.weight': src_state.pop(
        'body.6.block1.weight'), 'adapter.body.3.resnets.0.block1.bias':
        src_state.pop('body.6.block1.bias'),
        'adapter.body.3.resnets.0.block2.weight': src_state.pop(
        'body.6.block2.weight'), 'adapter.body.3.resnets.0.block2.bias':
        src_state.pop('body.6.block2.bias'),
        'adapter.body.3.resnets.1.block1.weight': src_state.pop(
        'body.7.block1.weight'), 'adapter.body.3.resnets.1.block1.bias':
        src_state.pop('body.7.block1.bias'),
        'adapter.body.3.resnets.1.block2.weight': src_state.pop(
        'body.7.block2.weight'), 'adapter.body.3.resnets.1.block2.bias':
        src_state.pop('body.7.block2.bias')}
    assert len(src_state) == 0
    adapter = T2IAdapter(in_channels=in_channels, adapter_type='full_adapter')
    adapter.load_state_dict(res_state)
    return adapter
