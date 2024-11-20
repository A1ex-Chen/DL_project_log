def hook_forwards(root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if 'attn2' in name and module.__class__.__name__ == 'Attention':
            module.forward = hook_forward(module)
