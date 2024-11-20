def convert_light_adapter(src_state):
    original_body_length = max([int(x.split('.')[1]) for x in src_state.
        keys() if 'body.' in x]) + 1
    assert original_body_length == 4
    res_state = {'adapter.body.0.in_conv.weight': src_state.pop(
        'body.0.in_conv.weight'), 'adapter.body.0.in_conv.bias': src_state.
        pop('body.0.in_conv.bias'),
        'adapter.body.0.resnets.0.block1.weight': src_state.pop(
        'body.0.body.0.block1.weight'),
        'adapter.body.0.resnets.0.block1.bias': src_state.pop(
        'body.0.body.0.block1.bias'),
        'adapter.body.0.resnets.0.block2.weight': src_state.pop(
        'body.0.body.0.block2.weight'),
        'adapter.body.0.resnets.0.block2.bias': src_state.pop(
        'body.0.body.0.block2.bias'),
        'adapter.body.0.resnets.1.block1.weight': src_state.pop(
        'body.0.body.1.block1.weight'),
        'adapter.body.0.resnets.1.block1.bias': src_state.pop(
        'body.0.body.1.block1.bias'),
        'adapter.body.0.resnets.1.block2.weight': src_state.pop(
        'body.0.body.1.block2.weight'),
        'adapter.body.0.resnets.1.block2.bias': src_state.pop(
        'body.0.body.1.block2.bias'),
        'adapter.body.0.resnets.2.block1.weight': src_state.pop(
        'body.0.body.2.block1.weight'),
        'adapter.body.0.resnets.2.block1.bias': src_state.pop(
        'body.0.body.2.block1.bias'),
        'adapter.body.0.resnets.2.block2.weight': src_state.pop(
        'body.0.body.2.block2.weight'),
        'adapter.body.0.resnets.2.block2.bias': src_state.pop(
        'body.0.body.2.block2.bias'),
        'adapter.body.0.resnets.3.block1.weight': src_state.pop(
        'body.0.body.3.block1.weight'),
        'adapter.body.0.resnets.3.block1.bias': src_state.pop(
        'body.0.body.3.block1.bias'),
        'adapter.body.0.resnets.3.block2.weight': src_state.pop(
        'body.0.body.3.block2.weight'),
        'adapter.body.0.resnets.3.block2.bias': src_state.pop(
        'body.0.body.3.block2.bias'), 'adapter.body.0.out_conv.weight':
        src_state.pop('body.0.out_conv.weight'),
        'adapter.body.0.out_conv.bias': src_state.pop(
        'body.0.out_conv.bias'), 'adapter.body.1.in_conv.weight': src_state
        .pop('body.1.in_conv.weight'), 'adapter.body.1.in_conv.bias':
        src_state.pop('body.1.in_conv.bias'),
        'adapter.body.1.resnets.0.block1.weight': src_state.pop(
        'body.1.body.0.block1.weight'),
        'adapter.body.1.resnets.0.block1.bias': src_state.pop(
        'body.1.body.0.block1.bias'),
        'adapter.body.1.resnets.0.block2.weight': src_state.pop(
        'body.1.body.0.block2.weight'),
        'adapter.body.1.resnets.0.block2.bias': src_state.pop(
        'body.1.body.0.block2.bias'),
        'adapter.body.1.resnets.1.block1.weight': src_state.pop(
        'body.1.body.1.block1.weight'),
        'adapter.body.1.resnets.1.block1.bias': src_state.pop(
        'body.1.body.1.block1.bias'),
        'adapter.body.1.resnets.1.block2.weight': src_state.pop(
        'body.1.body.1.block2.weight'),
        'adapter.body.1.resnets.1.block2.bias': src_state.pop(
        'body.1.body.1.block2.bias'),
        'adapter.body.1.resnets.2.block1.weight': src_state.pop(
        'body.1.body.2.block1.weight'),
        'adapter.body.1.resnets.2.block1.bias': src_state.pop(
        'body.1.body.2.block1.bias'),
        'adapter.body.1.resnets.2.block2.weight': src_state.pop(
        'body.1.body.2.block2.weight'),
        'adapter.body.1.resnets.2.block2.bias': src_state.pop(
        'body.1.body.2.block2.bias'),
        'adapter.body.1.resnets.3.block1.weight': src_state.pop(
        'body.1.body.3.block1.weight'),
        'adapter.body.1.resnets.3.block1.bias': src_state.pop(
        'body.1.body.3.block1.bias'),
        'adapter.body.1.resnets.3.block2.weight': src_state.pop(
        'body.1.body.3.block2.weight'),
        'adapter.body.1.resnets.3.block2.bias': src_state.pop(
        'body.1.body.3.block2.bias'), 'adapter.body.1.out_conv.weight':
        src_state.pop('body.1.out_conv.weight'),
        'adapter.body.1.out_conv.bias': src_state.pop(
        'body.1.out_conv.bias'), 'adapter.body.2.in_conv.weight': src_state
        .pop('body.2.in_conv.weight'), 'adapter.body.2.in_conv.bias':
        src_state.pop('body.2.in_conv.bias'),
        'adapter.body.2.resnets.0.block1.weight': src_state.pop(
        'body.2.body.0.block1.weight'),
        'adapter.body.2.resnets.0.block1.bias': src_state.pop(
        'body.2.body.0.block1.bias'),
        'adapter.body.2.resnets.0.block2.weight': src_state.pop(
        'body.2.body.0.block2.weight'),
        'adapter.body.2.resnets.0.block2.bias': src_state.pop(
        'body.2.body.0.block2.bias'),
        'adapter.body.2.resnets.1.block1.weight': src_state.pop(
        'body.2.body.1.block1.weight'),
        'adapter.body.2.resnets.1.block1.bias': src_state.pop(
        'body.2.body.1.block1.bias'),
        'adapter.body.2.resnets.1.block2.weight': src_state.pop(
        'body.2.body.1.block2.weight'),
        'adapter.body.2.resnets.1.block2.bias': src_state.pop(
        'body.2.body.1.block2.bias'),
        'adapter.body.2.resnets.2.block1.weight': src_state.pop(
        'body.2.body.2.block1.weight'),
        'adapter.body.2.resnets.2.block1.bias': src_state.pop(
        'body.2.body.2.block1.bias'),
        'adapter.body.2.resnets.2.block2.weight': src_state.pop(
        'body.2.body.2.block2.weight'),
        'adapter.body.2.resnets.2.block2.bias': src_state.pop(
        'body.2.body.2.block2.bias'),
        'adapter.body.2.resnets.3.block1.weight': src_state.pop(
        'body.2.body.3.block1.weight'),
        'adapter.body.2.resnets.3.block1.bias': src_state.pop(
        'body.2.body.3.block1.bias'),
        'adapter.body.2.resnets.3.block2.weight': src_state.pop(
        'body.2.body.3.block2.weight'),
        'adapter.body.2.resnets.3.block2.bias': src_state.pop(
        'body.2.body.3.block2.bias'), 'adapter.body.2.out_conv.weight':
        src_state.pop('body.2.out_conv.weight'),
        'adapter.body.2.out_conv.bias': src_state.pop(
        'body.2.out_conv.bias'), 'adapter.body.3.in_conv.weight': src_state
        .pop('body.3.in_conv.weight'), 'adapter.body.3.in_conv.bias':
        src_state.pop('body.3.in_conv.bias'),
        'adapter.body.3.resnets.0.block1.weight': src_state.pop(
        'body.3.body.0.block1.weight'),
        'adapter.body.3.resnets.0.block1.bias': src_state.pop(
        'body.3.body.0.block1.bias'),
        'adapter.body.3.resnets.0.block2.weight': src_state.pop(
        'body.3.body.0.block2.weight'),
        'adapter.body.3.resnets.0.block2.bias': src_state.pop(
        'body.3.body.0.block2.bias'),
        'adapter.body.3.resnets.1.block1.weight': src_state.pop(
        'body.3.body.1.block1.weight'),
        'adapter.body.3.resnets.1.block1.bias': src_state.pop(
        'body.3.body.1.block1.bias'),
        'adapter.body.3.resnets.1.block2.weight': src_state.pop(
        'body.3.body.1.block2.weight'),
        'adapter.body.3.resnets.1.block2.bias': src_state.pop(
        'body.3.body.1.block2.bias'),
        'adapter.body.3.resnets.2.block1.weight': src_state.pop(
        'body.3.body.2.block1.weight'),
        'adapter.body.3.resnets.2.block1.bias': src_state.pop(
        'body.3.body.2.block1.bias'),
        'adapter.body.3.resnets.2.block2.weight': src_state.pop(
        'body.3.body.2.block2.weight'),
        'adapter.body.3.resnets.2.block2.bias': src_state.pop(
        'body.3.body.2.block2.bias'),
        'adapter.body.3.resnets.3.block1.weight': src_state.pop(
        'body.3.body.3.block1.weight'),
        'adapter.body.3.resnets.3.block1.bias': src_state.pop(
        'body.3.body.3.block1.bias'),
        'adapter.body.3.resnets.3.block2.weight': src_state.pop(
        'body.3.body.3.block2.weight'),
        'adapter.body.3.resnets.3.block2.bias': src_state.pop(
        'body.3.body.3.block2.bias'), 'adapter.body.3.out_conv.weight':
        src_state.pop('body.3.out_conv.weight'),
        'adapter.body.3.out_conv.bias': src_state.pop('body.3.out_conv.bias')}
    assert len(src_state) == 0
    adapter = T2IAdapter(in_channels=3, channels=[320, 640, 1280],
        num_res_blocks=4, adapter_type='light_adapter')
    adapter.load_state_dict(res_state)
    return adapter
