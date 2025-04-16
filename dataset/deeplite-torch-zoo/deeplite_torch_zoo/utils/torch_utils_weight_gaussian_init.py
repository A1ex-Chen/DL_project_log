def weight_gaussian_init(model: torch.nn.Module):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            else:
                continue
    return model
