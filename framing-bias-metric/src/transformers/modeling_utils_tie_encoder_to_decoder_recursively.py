def tie_encoder_to_decoder_recursively(decoder_pointer: nn.Module,
    encoder_pointer: nn.Module, module_name: str,
    uninitialized_encoder_weights: List[str], depth=0):
    assert isinstance(decoder_pointer, nn.Module) and isinstance(
        encoder_pointer, nn.Module
        ), f'{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module'
    if hasattr(decoder_pointer, 'weight'):
        assert hasattr(encoder_pointer, 'weight')
        encoder_pointer.weight = decoder_pointer.weight
        if hasattr(decoder_pointer, 'bias'):
            assert hasattr(encoder_pointer, 'bias')
            encoder_pointer.bias = decoder_pointer.bias
        return
    encoder_modules = encoder_pointer._modules
    decoder_modules = decoder_pointer._modules
    if len(decoder_modules) > 0:
        assert len(encoder_modules
            ) > 0, f'Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}'
        all_encoder_weights = set([(module_name + '/' + sub_name) for
            sub_name in encoder_modules.keys()])
        encoder_layer_pos = 0
        for name, module in decoder_modules.items():
            if name.isdigit():
                encoder_name = str(int(name) + encoder_layer_pos)
                decoder_name = name
                if not isinstance(decoder_modules[decoder_name], type(
                    encoder_modules[encoder_name])) and len(encoder_modules
                    ) != len(decoder_modules):
                    encoder_layer_pos -= 1
                    continue
            elif name not in encoder_modules:
                continue
            elif depth > 500:
                raise ValueError(
                    'Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model.'
                    )
            else:
                decoder_name = encoder_name = name
            tie_encoder_to_decoder_recursively(decoder_modules[decoder_name
                ], encoder_modules[encoder_name], module_name + '/' + name,
                uninitialized_encoder_weights, depth=depth + 1)
            all_encoder_weights.remove(module_name + '/' + encoder_name)
        uninitialized_encoder_weights += list(all_encoder_weights)
