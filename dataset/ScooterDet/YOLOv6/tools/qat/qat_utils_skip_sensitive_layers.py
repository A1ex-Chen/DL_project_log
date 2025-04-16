def skip_sensitive_layers(model, sensitive_layers):
    print('Skip sensitive layers...')
    for name, module in model.named_modules():
        if name in sensitive_layers:
            print(f'Disable {name}')
            module_quant_disable(model, name)
